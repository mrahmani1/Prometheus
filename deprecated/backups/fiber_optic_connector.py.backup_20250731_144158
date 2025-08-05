"""Fiber optic sensing connector for DAS/DTS data"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, AsyncGenerator
import structlog
import h5py
import aiofiles
import os
from pathlib import Path

from .base_connector import BaseConnector
from models.geological_data import FiberOpticMeasurement, GeologicalLocation, StrainMeasurement

logger = structlog.get_logger()

class FiberOpticConnector(BaseConnector):
    """Connector for Distributed Acoustic Sensing (DAS) and Distributed Temperature Sensing (DTS) data"""
    
    def __init__(self, config: Any, data_processor: Any):
        super().__init__(config, data_processor)
        self.name = "fiber_optic"
        
        # DAS/DTS configuration
        self.das_sampling_rate = config.das_sampling_rate  # Hz
        self.dts_sampling_rate = config.dts_sampling_rate  # Hz
        self.fiber_length_km = config.fiber_length_km
        self.spatial_resolution = 1.0  # meters
        self.gauge_length = 10.0  # meters
        
        # Data storage paths
        self.data_directory = Path("/data/fiber_optic")
        self.das_directory = self.data_directory / "das"
        self.dts_directory = self.data_directory / "dts"
        
        # Processing parameters
        self.buffer_size = 1000  # Number of samples to buffer
        self.das_buffer = []
        self.dts_buffer = []
        
        # Simulated fiber coordinates (in practice, these would be surveyed)
        self.fiber_coordinates = self._generate_fiber_coordinates()
        
    async def initialize(self):
        """Initialize the fiber optic connector"""
        await super().initialize()
        
        # Create data directories
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.das_directory.mkdir(parents=True, exist_ok=True)
        self.dts_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "Fiber optic connector initialized",
            das_sampling_rate=self.das_sampling_rate,
            dts_sampling_rate=self.dts_sampling_rate,
            fiber_length_km=self.fiber_length_km
        )
    
    def _generate_fiber_coordinates(self) -> List[GeologicalLocation]:
        """Generate coordinates along the fiber cable"""
        coordinates = []
        
        # Simulate a fiber cable path (in practice, this would be surveyed data)
        # For this example, create a linear path
        start_lat, start_lon = 45.0, -120.0  # Example coordinates
        end_lat, end_lon = 45.1, -119.9
        
        num_points = int(self.fiber_length_km * 1000 / self.spatial_resolution)
        
        for i in range(num_points):
            fraction = i / (num_points - 1) if num_points > 1 else 0
            
            lat = start_lat + (end_lat - start_lat) * fraction
            lon = start_lon + (end_lon - start_lon) * fraction
            
            # Add some elevation variation
            elevation = 1000 + 200 * np.sin(fraction * np.pi * 4)
            
            coordinates.append(GeologicalLocation(
                latitude=lat,
                longitude=lon,
                elevation=elevation,
                depth=0.0  # Fiber is typically buried at shallow depth
            ))
        
        return coordinates
    
    async def _simulate_das_data(self, duration_seconds: int) -> np.ndarray:
        """Simulate DAS strain rate data"""
        
        num_channels = len(self.fiber_coordinates)
        num_samples = int(duration_seconds * self.das_sampling_rate)
        
        # Generate synthetic DAS data
        # In practice, this would read from actual DAS interrogator
        
        # Base noise level
        data = np.random.normal(0, 1e-9, (num_channels, num_samples))  # strain rate (1/s)
        
        # Add some coherent signals (simulated seismic events)
        for _ in range(np.random.poisson(2)):  # Random number of events
            # Random event parameters
            event_time = np.random.uniform(0, duration_seconds)
            event_channel = np.random.randint(0, num_channels)
            event_amplitude = np.random.uniform(1e-8, 1e-6)
            event_duration = np.random.uniform(1, 10)  # seconds
            
            # Create event waveform
            event_start_sample = int(event_time * self.das_sampling_rate)
            event_samples = int(event_duration * self.das_sampling_rate)
            
            if event_start_sample + event_samples < num_samples:
                # Simple Ricker wavelet
                t = np.linspace(0, event_duration, event_samples)
                f0 = 10.0  # Hz
                wavelet = event_amplitude * (1 - 2 * (np.pi * f0 * t)**2) * np.exp(-(np.pi * f0 * t)**2)
                
                # Add to multiple channels with distance-based attenuation
                for ch in range(max(0, event_channel - 50), min(num_channels, event_channel + 50)):
                    distance = abs(ch - event_channel) * self.spatial_resolution
                    attenuation = np.exp(-distance / 1000.0)  # 1km attenuation length
                    
                    end_sample = min(num_samples, event_start_sample + event_samples)
                    actual_samples = end_sample - event_start_sample
                    
                    data[ch, event_start_sample:end_sample] += wavelet[:actual_samples] * attenuation
        
        return data
    
    async def _simulate_dts_data(self, duration_seconds: int) -> np.ndarray:
        """Simulate DTS temperature data"""
        
        num_channels = len(self.fiber_coordinates)
        num_samples = int(duration_seconds * self.dts_sampling_rate)
        
        # Generate synthetic DTS data
        # In practice, this would read from actual DTS interrogator
        
        # Base temperature profile (varies with elevation)
        base_temps = []
        for coord in self.fiber_coordinates:
            # Temperature decreases with elevation (lapse rate ~6.5°C/km)
            temp = 20.0 - (coord.elevation / 1000.0) * 6.5
            base_temps.append(temp)
        
        base_temps = np.array(base_temps)
        
        # Create time series with small variations
        data = np.zeros((num_channels, num_samples))
        
        for i in range(num_samples):
            # Add temporal variations
            time_variation = 2.0 * np.sin(2 * np.pi * i / (self.dts_sampling_rate * 3600))  # Hourly cycle
            noise = np.random.normal(0, 0.1, num_channels)  # Measurement noise
            
            data[:, i] = base_temps + time_variation + noise
        
        return data
    
    async def process_das_data_chunk(
        self,
        data: np.ndarray,
        start_time: datetime,
        sampling_rate: float
    ) -> List[StrainMeasurement]:
        """Process a chunk of DAS data"""
        
        measurements = []
        num_channels, num_samples = data.shape
        
        # Process data in time windows
        window_size = int(sampling_rate)  # 1-second windows
        
        for sample_idx in range(0, num_samples, window_size):
            end_idx = min(sample_idx + window_size, num_samples)
            window_data = data[:, sample_idx:end_idx]
            
            # Calculate timestamp for this window
            window_time = start_time + timedelta(seconds=sample_idx / sampling_rate)
            
            # Process each channel
            for ch_idx in range(num_channels):
                if ch_idx < len(self.fiber_coordinates):
                    channel_data = window_data[ch_idx, :]
                    
                    # Calculate strain rate statistics
                    strain_rate_mean = np.mean(channel_data)
                    strain_rate_std = np.std(channel_data)
                    strain_rate_max = np.max(np.abs(channel_data))
                    
                    # Only create measurement if signal is above noise threshold
                    if strain_rate_max > 1e-8:  # Threshold for significant strain
                        measurement = StrainMeasurement(
                            timestamp=window_time,
                            location=self.fiber_coordinates[ch_idx],
                            strain_rate=strain_rate_mean,
                            strain_rate_std=strain_rate_std,
                            strain_rate_max=strain_rate_max,
                            measurement_type="das",
                            channel_index=ch_idx,
                            gauge_length=self.gauge_length,
                            sampling_rate=sampling_rate,
                            source="fiber_optic_das"
                        )
                        
                        measurements.append(measurement)
        
        return measurements
    
    async def process_dts_data_chunk(
        self,
        data: np.ndarray,
        start_time: datetime,
        sampling_rate: float
    ) -> List[FiberOpticMeasurement]:
        """Process a chunk of DTS data"""
        
        measurements = []
        num_channels, num_samples = data.shape
        
        # Process data in time windows
        window_size = max(1, int(sampling_rate))  # At least 1 sample
        
        for sample_idx in range(0, num_samples, window_size):
            end_idx = min(sample_idx + window_size, num_samples)
            window_data = data[:, sample_idx:end_idx]
            
            # Calculate timestamp for this window
            window_time = start_time + timedelta(seconds=sample_idx / sampling_rate)
            
            # Process each channel
            for ch_idx in range(num_channels):
                if ch_idx < len(self.fiber_coordinates):
                    channel_data = window_data[ch_idx, :]
                    
                    # Calculate temperature statistics
                    temperature_mean = np.mean(channel_data)
                    temperature_std = np.std(channel_data)
                    
                    measurement = FiberOpticMeasurement(
                        timestamp=window_time,
                        location=self.fiber_coordinates[ch_idx],
                        measurement_type="dts",
                        temperature=temperature_mean,
                        temperature_std=temperature_std,
                        channel_index=ch_idx,
                        spatial_resolution=self.spatial_resolution,
                        sampling_rate=sampling_rate,
                        source="fiber_optic_dts"
                    )
                    
                    measurements.append(measurement)
        
        return measurements
    
    async def ingest_data_range(
        self,
        start_time: datetime,
        end_time: datetime,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest fiber optic data for a specific time range"""
        
        params = parameters or {}
        results = {
            "connector": self.name,
            "start_time": start_time,
            "end_time": end_time,
            "records_processed": 0,
            "errors": []
        }
        
        try:
            duration_seconds = int((end_time - start_time).total_seconds())
            
            # Process DAS data if requested
            if params.get("include_das", True):
                logger.info("Processing DAS data", duration=duration_seconds)
                
                das_data = await self._simulate_das_data(duration_seconds)
                das_measurements = await self.process_das_data_chunk(
                    das_data, start_time, self.das_sampling_rate
                )
                
                for measurement in das_measurements:
                    try:
                        await self.data_processor.process_strain_measurement(measurement)
                        results["records_processed"] += 1
                    except Exception as e:
                        error_msg = f"Failed to process DAS measurement: {str(e)}"
                        results["errors"].append(error_msg)
                        logger.warning(error_msg)
            
            # Process DTS data if requested
            if params.get("include_dts", True):
                logger.info("Processing DTS data", duration=duration_seconds)
                
                dts_data = await self._simulate_dts_data(duration_seconds)
                dts_measurements = await self.process_dts_data_chunk(
                    dts_data, start_time, self.dts_sampling_rate
                )
                
                for measurement in dts_measurements:
                    try:
                        await self.data_processor.process_fiber_optic_measurement(measurement)
                        results["records_processed"] += 1
                    except Exception as e:
                        error_msg = f"Failed to process DTS measurement: {str(e)}"
                        results["errors"].append(error_msg)
                        logger.warning(error_msg)
            
            # Update statistics
            self.total_records_processed += results["records_processed"]
            self.last_data_time = datetime.utcnow()
            
            logger.info(
                "Fiber optic data ingestion completed",
                records_processed=results["records_processed"],
                errors=len(results["errors"])
            )
            
        except Exception as e:
            error_msg = f"Fiber optic data ingestion failed: {str(e)}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
            self.error_count += 1
        
        return results
    
    async def start_continuous_ingestion(self):
        """Start continuous data ingestion"""
        logger.info("Starting continuous fiber optic data ingestion")
        
        while not self.is_paused:
            try:
                # Process data in 10-second chunks for real-time processing
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(seconds=10)
                
                await self.ingest_data_range(start_time, end_time)
                
                # Wait before next ingestion cycle
                await asyncio.sleep(10)  # 10 seconds
                
            except asyncio.CancelledError:
                logger.info("Fiber optic continuous ingestion cancelled")
                break
            except Exception as e:
                logger.error("Error in continuous fiber optic ingestion", error=str(e))
                self.error_count += 1
                await asyncio.sleep(30)  # Wait 30 seconds before retrying
    
    async def get_fiber_geometry(self) -> List[Dict[str, Any]]:
        """Get fiber cable geometry information"""
        geometry = []
        
        for i, coord in enumerate(self.fiber_coordinates):
            geometry.append({
                "channel_index": i,
                "distance_m": i * self.spatial_resolution,
                "latitude": coord.latitude,
                "longitude": coord.longitude,
                "elevation": coord.elevation,
                "depth": coord.depth
            })
        
        return geometry
    
    async def cleanup(self):
        """Cleanup the fiber optic connector"""
        await super().cleanup()
        logger.info("Fiber optic connector cleanup completed")