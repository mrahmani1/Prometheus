"""
Alaska Satellite Facility (ASF) InSAR Pipeline Integration

This module provides seamless integration with ASF's InSAR processing pipeline
for advanced ground deformation analysis including:
    pass
- Automated SAR data discovery and download from ASF DAAC
- Integration with HyP3 cloud processing for InSAR generation
- Advanced deformation time series analysis
- Integration with Volcano Watchtower and Vulcan UI
- Real-time deformation monitoring and alerting
"""

import asyncio
import logging
import os
import signal
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import rasterio
import torch.nn.functional as F
from scipy import ndimage
from shapely.geometry import Point, Polygon, box

logger = logging.getLogger(__name__)

@dataclass
class ASFSearchParameters:
    """Parameters for ASF data search"""

    platform: str = "SENTINEL-1"  # SENTINEL-1, ALOS, RADARSAT-1, etc.
    instrument: str = "C-SAR"
    beam_mode: str = "IW"  # Interferometric Wide swath
    product_type: str = "SLC"  # Single Look Complex
    processing_level: str = "L1"

    # Temporal parameters
    start_date: datetime = None
    end_date: datetime = None

    # Spatial parameters
    geometry: Union[Polygon, Point] = None
    path: Optional[int] = None
    frame: Optional[int] = None

    # Processing parameters
    polarization: str = "VV"
    orbit_direction: str = "ASCENDING"  # or DESCENDING

    # Quality filters
    max_baseline_temporal: int = 48  # days
    max_baseline_perpendicular: float = 200.0  # meters
    min_coherence: float = 0.3

    def __post_init__(self):
        if self.start_date is None:
            self.start_date = datetime.utcnow() - timedelta(days=365)
        if self.end_date is None:
            self.end_date = datetime.utcnow()

@dataclass
class InSARProduct:
    """InSAR product metadata and data"""

    product_id: str
    granule_name: str
    platform: str
    acquisition_date: datetime
    orbit_direction: str
    path: int
    frame: int
    polarization: str

    # Geometric parameters
    center_lat: float
    center_lon: float
    bbox: Tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat

    # Processing parameters
    baseline_temporal: int  # days
    baseline_perpendicular: float  # meters
    coherence_mean: float

    # File paths
    interferogram_path: Optional[str] = None
    coherence_path: Optional[str] = None
    unwrapped_phase_path: Optional[str] = None
    displacement_path: Optional[str] = None

    # Processing status
    processing_status: str = "pending"
    processing_job_id: Optional[str] = None
    download_url: Optional[str] = None

    @property
    def is_processed(self) -> bool:
        return self.processing_status == "completed"

    @property
    def geometry(self) -> Polygon:
        """Get product geometry as Shapely polygon"""
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return box(min_lon, min_lat, max_lon, max_lat)

@dataclass
class DeformationTimeSeries:
    """Time series of ground deformation measurements"""

    location: Tuple[float, float]  # lat, lon
    timestamps: List[datetime]
    displacement_los: List[float]  # Line-of-sight displacement in mm
    displacement_east: List[float]  # East component in mm
    displacement_north: List[float]  # North component in mm
    displacement_up: List[float]  # Vertical component in mm

    coherence_values: List[float]
    uncertainty_mm: List[float]

    # Metadata
    reference_date: datetime
    reference_point: Tuple[float, float]
    processing_method: str = "SBAS"  # Small Baseline Subset

    @property
    def total_displacement(self) -> List[float]:
        """Calculate total 3D displacement magnitude"""
        return [
            np.sqrt(e**2 + n**2 + u**2)
            for e, n, u in zip(
                self.displacement_east, self.displacement_north, self.displacement_up
            )
        ]

    @property
    def velocity_mm_year(self) -> Tuple[float, float, float, float]:
        """Calculate average velocity in mm/year (east, north, up, total)"""
        if len(self.timestamps) < 2:
            return (0.0, 0.0, 0.0, 0.0)

        time_span_years = (self.timestamps[-1] - self.timestamps[0]).days / 365.25

        east_vel = (
            self.displacement_east[-1] - self.displacement_east[0]
        ) / time_span_years
        north_vel = (
            self.displacement_north[-1] - self.displacement_north[0]
        ) / time_span_years
        up_vel = (self.displacement_up[-1] - self.displacement_up[0]) / time_span_years
        total_vel = np.sqrt(east_vel**2 + north_vel**2 + up_vel**2)

        return (east_vel, north_vel, up_vel, total_vel)

class ASFDataClient:
    """Client for Alaska Satellite Facility data access"""

    def __init__(self, username: str = None, password: str = None):
        self.base_url = "https://api.daac.asf.alaska.edu"
        self.search_url = f"{self.base_url}/services/search/param"
        self.download_url = f"{self.base_url}/services/utils/files_to_metalink"

        # Authentication
        self.username = username or os.getenv("ASF_USERNAME")
        self.password = password or os.getenv("ASF_PASSWORD")

        if not self.username or not self.password:
            logger.warning(
                "ASF credentials not provided. Some features may be limited."
            )

        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        auth = (
            aiohttp.BasicAuth(self.username, self.password) if self.username else None
        )
        self.session = aiohttp.ClientSession(auth=auth)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search_products(
        self, search_params: ASFSearchParameters
    ) -> List[InSARProduct]:
        """Search for SAR products matching criteria"""

        # Build search parameters
        params = {
            "platform": search_params.platform,
            "instrument": search_params.instrument,
            "beamMode": search_params.beam_mode,
            "productType": search_params.product_type,
            "processingLevel": search_params.processing_level,
            "start": search_params.start_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "end": search_params.end_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "output": "jsonlite",
        }

        # Add spatial parameters
        if search_params.geometry:
            if isinstance(search_params.geometry, Point):
                params["point"] = (
                    f"{search_params.geometry.y},{search_params.geometry.x}"
                )
            elif isinstance(search_params.geometry, Polygon):
                # Convert polygon to WKT
                params["polygon"] = search_params.geometry.wkt

        if search_params.path:
            params["relativeOrbit"] = search_params.path
        if search_params.frame:
            params["frame"] = search_params.frame

        # Add processing parameters
        if search_params.polarization:
            params["polarization"] = search_params.polarization
        if search_params.orbit_direction:
            params["flightDirection"] = search_params.orbit_direction

        try:
            async with self.session.get(self.search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_search_results(data)
                else:
                    logger.error(f"ASF search failed: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Failed to search ASF products: {e}")
            return []

    def _parse_search_results(self, search_data: Dict) -> List[InSARProduct]:
        """Parse ASF search results into InSARProduct objects"""

        products = []

        for item in search_data.get("results", []):
            try:
                # Parse acquisition date
                acq_date = datetime.strptime(item["startTime"], "%Y-%m-%dT%H:%M:%S")

                # Extract geometric information
                coords = item.get("polygon", "")
                bbox = self._extract_bbox_from_polygon(coords)
                center_lat = (bbox[1] + bbox[3]) / 2
                center_lon = (bbox[0] + bbox[2]) / 2

                product = InSARProduct(
                    product_id=item["granuleName"],
                    granule_name=item["granuleName"],
                    platform=item["platform"],
                    acquisition_date=acq_date,
                    orbit_direction=item.get("flightDirection", "UNKNOWN"),
                    path=int(item.get("pathNumber", 0)),
                    frame=int(item.get("frameNumber", 0)),
                    polarization=item.get("polarization", "VV"),
                    center_lat=center_lat,
                    center_lon=center_lon,
                    bbox=bbox,
                    baseline_temporal=0,  # Will be calculated during pair selection
                    baseline_perpendicular=0.0,  # Will be calculated during pair selection
                    coherence_mean=0.0,  # Will be calculated after processing
                    download_url=item.get("url", ""),
                )

                products.append(product)

            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse search result: {e}")
                continue

        return products

    def _extract_bbox_from_polygon(
        self, polygon_wkt: str
    ) -> Tuple[float, float, float, float]:
        """Extract bounding box from polygon WKT string"""

        try:
            # Simple parsing for POLYGON((lon lat, ...)) format
            coords_str = polygon_wkt.replace("POLYGON((", "").replace("))", "")
            coord_pairs = coords_str.split(", ")

            lons = []
            lats = []

            for pair in coord_pairs:
                lon, lat = map(float, pair.split())
                lons.append(lon)
                lats.append(lat)

            return (min(lons), min(lats), max(lons), max(lats))

        except Exception:
            # Return default bbox if parsing fails
            return (-180.0, -90.0, 180.0, 90.0)

    async def download_product(self, product: InSARProduct, download_dir: str) -> str:
        """Download SAR product from ASF"""

        download_path = Path(download_dir) / f"{product.granule_name}.zip"

        try:
            async with self.session.get(product.download_url) as response:
                if response.status == 200:
                    with open(download_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)

                    logger.info(f"Downloaded {product.granule_name} to {download_path}")
                    return str(download_path)
                else:
                    logger.error(
                        f"Download failed for {product.granule_name}: {response.status}"
                    )
                    return None
        except Exception as e:
            logger.error(f"Failed to download {product.granule_name}: {e}")
            return None

class HyP3ProcessingClient:
    """Client for ASF HyP3 cloud processing service"""

    def __init__(self, username: str = None, password: str = None):
        self.base_url = "https://hyp3-api.asf.alaska.edu"
        self.username = username or os.getenv("ASF_USERNAME")
        self.password = password or os.getenv("ASF_PASSWORD")
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        auth = (
            aiohttp.BasicAuth(self.username, self.password) if self.username else None
        )
        self.session = aiohttp.ClientSession(auth=auth)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def submit_insar_job(
        self, reference_scene: str, secondary_scene: str, job_name: str = None
    ) -> Optional[str]:
        """Submit InSAR processing job to HyP3"""

        job_data = {
            "job_type": "INSAR_GAMMA",
            "job_parameters": {
                "granules": [reference_scene, secondary_scene],
                "include_look_vectors": True,
                "include_los_displacement": True,
                "include_inc_map": True,
                "apply_water_mask": True,
                "include_wrapped_phase": True,
                "include_displacement_maps": True,
            },
        }

        if job_name:
            job_data["name"] = job_name

        try:
            async with self.session.post(
                f"{self.base_url}/jobs", json=job_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    job_id = result.get("jobs", [{}])[0].get("job_id")
                    logger.info(f"Submitted InSAR job: {job_id}")
                    return job_id
                else:
                    logger.error(f"Failed to submit InSAR job: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error submitting InSAR job: {e}")
            return None

    async def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check status of HyP3 processing job"""

        try:
            async with self.session.get(f"{self.base_url}/jobs/{job_id}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to check job status: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            return {}

    async def download_job_results(self, job_id: str, download_dir: str) -> List[str]:
        """Download results from completed HyP3 job"""

        job_status = await self.check_job_status(job_id)

        if job_status.get("status_code") != "SUCCEEDED":
            logger.warning(f"Job {job_id} not completed successfully")
            return []

        download_urls = job_status.get("files", [])
        downloaded_files = []

        for file_info in download_urls:
            file_url = file_info.get("url", "")
            filename = file_info.get("filename", f"hyp3_result_{len(downloaded_files)}")

            download_path = Path(download_dir) / filename

            try:
                async with self.session.get(file_url) as response:
                    if response.status == 200:
                        with open(download_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)

                        downloaded_files.append(str(download_path))
                        logger.info(f"Downloaded {filename}")
                    else:
                        logger.error(
                            f"Failed to download {filename}: {response.status}"
                        )
            except Exception as e:
                logger.error(f"Error downloading {filename}: {e}")

        return downloaded_files

class AdvancedInSARProcessor:
    """Advanced InSAR processing and analysis"""

    def __init__(self):
        self.reference_point: Optional[Tuple[float, float]] = None
        self.processing_parameters = {
            "coherence_threshold": 0.3,
            "phase_unwrapping_method": "snaphu",
            "atmospheric_correction": True,
            "topographic_correction": True,
            "temporal_filtering": True,
            "spatial_filtering": True,
        }

    def process_interferogram(
        self, interferogram_path: str, coherence_path: str, output_dir: str
    ) -> Dict[str, str]:
        """Process interferogram to generate displacement maps"""

        results = {
            "unwrapped_phase": None,
            "displacement_los": None,
            "displacement_east": None,
            "displacement_north": None,
            "displacement_up": None,
            "quality_mask": None,
        }

        try:
            # Load interferogram and coherence
            with rasterio.open(interferogram_path) as ifg_src:
                interferogram = ifg_src.read(1)
                transform = ifg_src.transform
                crs = ifg_src.crs

            with rasterio.open(coherence_path) as coh_src:
                coherence = coh_src.read(1)

            # Quality mask based on coherence
            quality_mask = (
                coherence >= self.processing_parameters["coherence_threshold"]
            )

            # Phase unwrapping (simplified - would use SNAPHU in practice)
            unwrapped_phase = self._unwrap_phase(interferogram, quality_mask)

            # Convert phase to displacement
            wavelength = 0.056  # Sentinel-1 C-band wavelength in meters
            displacement_los = (
                unwrapped_phase * wavelength / (4 * np.pi) * 1000
            )  # Convert to mm

            # Apply atmospheric correction (simplified)
            if self.processing_parameters["atmospheric_correction"]:
                displacement_los = self._apply_atmospheric_correction(
                    displacement_los, quality_mask
                )

            # Decompose LOS displacement to 3D components (requires ascending/descending pairs)
            displacement_east, displacement_north, displacement_up = (
                self._decompose_los_displacement(
                    displacement_los, orbit_direction="ASCENDING"
                )
            )

            # Save results
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save unwrapped phase
            unwrapped_path = output_path / "unwrapped_phase.ti"
            self._save_raster(unwrapped_phase, unwrapped_path, transform, crs)
            results["unwrapped_phase"] = str(unwrapped_path)

            # Save displacement components
            disp_los_path = output_path / "displacement_los.ti"
            self._save_raster(displacement_los, disp_los_path, transform, crs)
            results["displacement_los"] = str(disp_los_path)

            disp_east_path = output_path / "displacement_east.ti"
            self._save_raster(displacement_east, disp_east_path, transform, crs)
            results["displacement_east"] = str(disp_east_path)

            disp_north_path = output_path / "displacement_north.ti"
            self._save_raster(displacement_north, disp_north_path, transform, crs)
            results["displacement_north"] = str(disp_north_path)

            disp_up_path = output_path / "displacement_up.ti"
            self._save_raster(displacement_up, disp_up_path, transform, crs)
            results["displacement_up"] = str(disp_up_path)

            # Save quality mask
            quality_path = output_path / "quality_mask.ti"
            self._save_raster(
                quality_mask.astype(np.uint8), quality_path, transform, crs
            )
            results["quality_mask"] = str(quality_path)

            logger.info(f"Processed interferogram: {interferogram_path}")

        except Exception as e:
            logger.error(f"Failed to process interferogram: {e}")

        return results

    def _unwrap_phase(
        self, wrapped_phase: np.ndarray, quality_mask: np.ndarray
    ) -> np.ndarray:
        """Phase unwrapping using simplified algorithm"""

        # Mask low-quality areas
        masked_phase = wrapped_phase.copy()
        masked_phase[~quality_mask] = 0

        # Simple phase unwrapping using scipy (in practice, would use SNAPHU)
        unwrapped = np.unwrap(np.unwrap(masked_phase, axis=0), axis=1)

        # Apply quality mask
        unwrapped[~quality_mask] = np.nan

        return unwrapped

    def _apply_atmospheric_correction(
        self, displacement: np.ndarray, quality_mask: np.ndarray
    ) -> np.ndarray:
        """Apply atmospheric phase screen correction"""

        # Simplified atmospheric correction using spatial filtering
        # In practice, would use weather models or empirical methods

        # Create atmospheric phase screen estimate
        valid_data = displacement[quality_mask]
        if len(valid_data) == 0:
            return displacement

        # Remove long-wavelength atmospheric signals
        corrected = displacement.copy()

        # Apply Gaussian filter to estimate atmospheric component
        atmospheric_component = ndimage.gaussian_filter(
            displacement, sigma=50, mode="constant", cval=0
        )

        # Subtract atmospheric component
        corrected -= atmospheric_component

        return corrected

    def _decompose_los_displacement(
        self, los_displacement: np.ndarray, orbit_direction: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose line-of-sight displacement to 3D components"""

        # Simplified decomposition (requires multiple viewing geometries for full 3D)
        # This is a basic approximation

        if orbit_direction == "ASCENDING":
            # Typical Sentinel-1 ascending geometry
            incidence_angle = np.radians(39)  # degrees
            heading_angle = np.radians(-13)  # degrees
        else:  # DESCENDING
            incidence_angle = np.radians(39)
            heading_angle = np.radians(193)

        # Unit vector components for LOS
        np.sin(incidence_angle) * np.sin(heading_angle)
        np.sin(incidence_angle) * np.cos(heading_angle)
        los_up = np.cos(incidence_angle)

        # For single LOS, assume horizontal motion is minimal
        # This is a simplified approach - proper 3D decomposition needs multiple geometries
        displacement_up = los_displacement / los_up
        displacement_east = np.zeros_like(los_displacement)
        displacement_north = np.zeros_like(los_displacement)

        return displacement_east, displacement_north, displacement_up

    def _save_raster(
        self, data: np.ndarray, output_path: Path, transform, crs, dtype=None
    ):
        """Save numpy array as GeoTIFF"""

        if dtype is None:
            dtype = data.dtype

        with rasterio.open(
            output_path,
            "w",
            driver="GTi",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(data, 1)

    def generate_time_series(
        self,
        displacement_files: List[str],
        timestamps: List[datetime],
        reference_point: Tuple[float, float],
    ) -> DeformationTimeSeries:
        """Generate deformation time series for a specific location"""

        self.reference_point = reference_point

        # Extract displacement values at reference point
        los_displacements = []
        east_displacements = []
        north_displacements = []
        up_displacements = []
        coherence_values = []
        uncertainties = []

        for i, disp_file in enumerate(displacement_files):
            try:
                with rasterio.open(disp_file) as src:
                    # Convert lat/lon to pixel coordinates
                    row, col = src.index(reference_point[1], reference_point[0])

                    # Extract displacement value
                    displacement = src.read(1)[row, col]

                    # Determine displacement component from filename
                    if "los" in disp_file.lower():
                        los_displacements.append(displacement)
                    elif "east" in disp_file.lower():
                        east_displacements.append(displacement)
                    elif "north" in disp_file.lower():
                        north_displacements.append(displacement)
                    elif "up" in disp_file.lower():
                        up_displacements.append(displacement)

                # Estimate uncertainty (simplified)
                uncertainties.append(2.0)  # 2mm uncertainty
                coherence_values.append(0.8)  # Mock coherence

            except Exception as e:
                logger.warning(f"Failed to extract displacement from {disp_file}: {e}")
                los_displacements.append(0.0)
                east_displacements.append(0.0)
                north_displacements.append(0.0)
                up_displacements.append(0.0)
                uncertainties.append(5.0)
                coherence_values.append(0.3)

        # Ensure all lists have same length
        n_points = len(timestamps)
        los_displacements = los_displacements[:n_points] + [0.0] * (
            n_points - len(los_displacements)
        )
        east_displacements = east_displacements[:n_points] + [0.0] * (
            n_points - len(east_displacements)
        )
        north_displacements = north_displacements[:n_points] + [0.0] * (
            n_points - len(north_displacements)
        )
        up_displacements = up_displacements[:n_points] + [0.0] * (
            n_points - len(up_displacements)
        )

        return DeformationTimeSeries(
            location=reference_point,
            timestamps=timestamps,
            displacement_los=los_displacements,
            displacement_east=east_displacements,
            displacement_north=north_displacements,
            displacement_up=up_displacements,
            coherence_values=coherence_values,
            uncertainty_mm=uncertainties,
            reference_date=timestamps[0] if timestamps else datetime.utcnow(),
            reference_point=reference_point,
            processing_method="ASF_HyP3_SBAS",
        )

class ASFInSARPipeline:
    """Complete ASF InSAR processing pipeline"""

    def __init__(self, asf_username: str = None, asf_password: str = None):
        self.asf_client = ASFDataClient(asf_username, asf_password)
        self.hyp3_client = HyP3ProcessingClient(asf_username, asf_password)
        self.processor = AdvancedInSARProcessor()

        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_products: List[InSARProduct] = []
        self.time_series_data: Dict[str, DeformationTimeSeries] = {}

    async def process_volcano_deformation(
        self,
        volcano_location: Tuple[float, float],
        volcano_name: str,
        start_date: datetime,
        end_date: datetime,
        output_dir: str,
    ) -> Dict[str, Any]:
        """Complete volcano deformation analysis pipeline"""

        results = {
            "volcano_name": volcano_name,
            "location": volcano_location,
            "processing_start": datetime.utcnow(),
            "search_results": [],
            "processing_jobs": [],
            "time_series": None,
            "deformation_analysis": {},
            "alerts": [],
        }

        try:
            # Step 1: Search for SAR data
            logger.info(f"Searching for SAR data around {volcano_name}")

            # Create search area (20km radius around volcano)
            lat, lon = volcano_location
            search_radius_deg = 0.18  # ~20km at equator
            search_geometry = Point(lon, lat).buffer(search_radius_deg)

            search_params = ASFSearchParameters(
                geometry=search_geometry,
                start_date=start_date,
                end_date=end_date,
                platform="SENTINEL-1",
                product_type="SLC",
                orbit_direction="ASCENDING",
            )

            async with self.asf_client:
                products = await self.asf_client.search_products(search_params)

            results["search_results"] = [
                {
                    "granule_name": p.granule_name,
                    "acquisition_date": p.acquisition_date.isoformat(),
                    "path": p.path,
                    "frame": p.frame,
                }
                for p in products
            ]

            logger.info(f"Found {len(products)} SAR products for {volcano_name}")

            # Step 2: Select interferometric pairs
            pairs = self._select_interferometric_pairs(products)
            logger.info(f"Selected {len(pairs)} interferometric pairs")

            # Step 3: Submit processing jobs to HyP3
            async with self.hyp3_client:
                for i, (ref_product, sec_product) in enumerate(
                    pairs[:5]
                ):  # Limit to 5 pairs
                    job_name = f"{volcano_name}_InSAR_{i+1}"

                    job_id = await self.hyp3_client.submit_insar_job(
                        ref_product.granule_name, sec_product.granule_name, job_name
                    )

                    if job_id:
                        self.active_jobs[job_id] = {
                            "volcano_name": volcano_name,
                            "reference_product": ref_product,
                            "secondary_product": sec_product,
                            "submitted_at": datetime.utcnow(),
                            "status": "submitted",
                        }

                        results["processing_jobs"].append(
                            {
                                "job_id": job_id,
                                "job_name": job_name,
                                "reference_scene": ref_product.granule_name,
                                "secondary_scene": sec_product.granule_name,
                            }
                        )

            # Step 4: Monitor job completion and process results
            completed_jobs = await self._monitor_and_process_jobs(
                volcano_name, output_dir
            )

            # Step 5: Generate time series analysis
            if completed_jobs:
                time_series = await self._generate_volcano_time_series(
                    volcano_location, completed_jobs, output_dir
                )
                results["time_series"] = time_series

                # Step 6: Analyze deformation patterns
                deformation_analysis = self._analyze_deformation_patterns(time_series)
                results["deformation_analysis"] = deformation_analysis

                # Step 7: Generate alerts if significant deformation detected
                alerts = self._generate_deformation_alerts(
                    deformation_analysis, volcano_name
                )
                results["alerts"] = alerts

            results["processing_end"] = datetime.utcnow()
            results["status"] = "completed"

        except Exception as e:
            logger.error(
                f"Failed to process volcano deformation for {volcano_name}: {e}"
            )
            results["status"] = "failed"
            results["error"] = str(e)

        return results

    def _select_interferometric_pairs(
        self, products: List[InSARProduct]
    ) -> List[Tuple[InSARProduct, InSARProduct]]:
        """Select optimal interferometric pairs based on temporal and spatial baselines"""

        pairs = []

        # Sort products by acquisition date
        sorted_products = sorted(products, key=lambda p: p.acquisition_date)

        for i, ref_product in enumerate(sorted_products):
            for j, sec_product in enumerate(sorted_products[i + 1 :], i + 1):
                # Calculate temporal baseline
                temporal_baseline = (
                    sec_product.acquisition_date - ref_product.acquisition_date
                ).days

                # Skip if temporal baseline is too large
                if temporal_baseline > 48:  # 48 days max
                    continue

                # Calculate perpendicular baseline (simplified - would need precise orbits)
                perp_baseline = (
                    abs(ref_product.path - sec_product.path) * 10
                )  # Rough estimate

                # Skip if perpendicular baseline is too large
                if perp_baseline > 200:  # 200m max
                    continue

                # Check if same path/frame
                if (
                    ref_product.path != sec_product.path
                    or ref_product.frame != sec_product.frame
                ):
                    continue

                pairs.append((ref_product, sec_product))

        # Sort pairs by temporal baseline (prefer shorter baselines)
        pairs.sort(key=lambda p: (p[1].acquisition_date - p[0].acquisition_date).days)

        return pairs

    async def _monitor_and_process_jobs(
        self, volcano_name: str, output_dir: str
    ) -> List[Dict[str, Any]]:
        """Monitor HyP3 jobs and process completed results"""

        completed_jobs = []
        max_wait_time = 3600  # 1 hour max wait
        check_interval = 60  # Check every minute
        elapsed_time = 0

        async with self.hyp3_client:
            while self.active_jobs and elapsed_time < max_wait_time:
                for job_id, job_info in list(self.active_jobs.items()):
                    if job_info["status"] == "completed":
                        continue

                    # Check job status
                    status = await self.hyp3_client.check_job_status(job_id)

                    if status.get("status_code") == "SUCCEEDED":
                        logger.info(f"Job {job_id} completed successfully")

                        # Download results
                        job_output_dir = Path(output_dir) / volcano_name / job_id
                        job_output_dir.mkdir(parents=True, exist_ok=True)

                        downloaded_files = await self.hyp3_client.download_job_results(
                            job_id, str(job_output_dir)
                        )

                        if downloaded_files:
                            # Process interferogram
                            processed_results = self._process_job_results(
                                downloaded_files, str(job_output_dir)
                            )

                            completed_job = {
                                "job_id": job_id,
                                "job_info": job_info,
                                "downloaded_files": downloaded_files,
                                "processed_results": processed_results,
                            }

                            completed_jobs.append(completed_job)
                            job_info["status"] = "completed"

                    elif status.get("status_code") == "FAILED":
                        logger.error(f"Job {job_id} failed")
                        job_info["status"] = "failed"

                await asyncio.sleep(check_interval)
                elapsed_time += check_interval

        return completed_jobs

    def _process_job_results(
        self, downloaded_files: List[str], output_dir: str
    ) -> Dict[str, str]:
        """Process downloaded HyP3 results"""

        # Find interferogram and coherence files
        interferogram_file = None
        coherence_file = None

        for file_path in downloaded_files:
            filename = Path(file_path).name.lower()
            if "unw_phase" in filename or "interferogram" in filename:
                interferogram_file = file_path
            elif "corr" in filename or "coherence" in filename:
                coherence_file = file_path

        if not interferogram_file or not coherence_file:
            logger.warning("Could not find interferogram or coherence files")
            return {}

        # Process interferogram
        processed_results = self.processor.process_interferogram(
            interferogram_file, coherence_file, output_dir
        )

        return processed_results

    async def _generate_volcano_time_series(
        self,
        volcano_location: Tuple[float, float],
        completed_jobs: List[Dict[str, Any]],
        output_dir: str,
    ) -> Dict[str, Any]:
        """Generate deformation time series for volcano"""

        # Collect displacement files and timestamps
        displacement_files = []
        timestamps = []

        for job in completed_jobs:
            processed_results = job.get("processed_results", {})
            job_info = job.get("job_info", {})

            if "displacement_los" in processed_results:
                displacement_files.append(processed_results["displacement_los"])

                # Use secondary scene date as timestamp
                sec_product = job_info.get("secondary_product")
                if sec_product:
                    timestamps.append(sec_product.acquisition_date)

        if not displacement_files:
            logger.warning("No displacement files found for time series generation")
            return {}

        # Generate time series
        time_series = self.processor.generate_time_series(
            displacement_files, timestamps, volcano_location
        )

        # Store time series data
        volcano_key = f"{volcano_location[0]:.4f}_{volcano_location[1]:.4f}"
        self.time_series_data[volcano_key] = time_series

        # Convert to serializable format
        return {
            "location": time_series.location,
            "timestamps": [t.isoformat() for t in time_series.timestamps],
            "displacement_los": time_series.displacement_los,
            "displacement_east": time_series.displacement_east,
            "displacement_north": time_series.displacement_north,
            "displacement_up": time_series.displacement_up,
            "total_displacement": time_series.total_displacement,
            "velocity_mm_year": time_series.velocity_mm_year,
            "coherence_values": time_series.coherence_values,
            "uncertainty_mm": time_series.uncertainty_mm,
            "reference_date": time_series.reference_date.isoformat(),
            "processing_method": time_series.processing_method,
        }

    def _analyze_deformation_patterns(
        self, time_series: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze deformation patterns for volcanic interpretation"""

        if not time_series:
            return {}

        analysis = {
            "summary_statistics": {},
            "trend_analysis": {},
            "anomaly_detection": {},
            "volcanic_interpretation": {},
        }

        # Extract displacement data
        up_displacement = time_series.get("displacement_up", [])
        total_displacement = time_series.get("total_displacement", [])
        timestamps = time_series.get("timestamps", [])

        if not up_displacement or len(up_displacement) < 2:
            return analysis

        # Summary statistics
        analysis["summary_statistics"] = {
            "max_uplift_mm": max(up_displacement),
            "max_subsidence_mm": min(up_displacement),
            "total_range_mm": max(up_displacement) - min(up_displacement),
            "mean_displacement_mm": np.mean(up_displacement),
            "std_displacement_mm": np.std(up_displacement),
            "max_total_displacement_mm": (
                max(total_displacement) if total_displacement else 0
            ),
        }

        # Trend analysis
        if len(up_displacement) >= 3:
            # Linear trend
            time_days = [
                (datetime.fromisoformat(t) - datetime.fromisoformat(timestamps[0])).days
                for t in timestamps
            ]

            # Simple linear regression
            A = np.vstack([time_days, np.ones(len(time_days))]).T
            slope, intercept = np.linalg.lstsq(A, up_displacement, rcond=None)[0]

            analysis["trend_analysis"] = {
                "linear_trend_mm_day": slope,
                "linear_trend_mm_year": slope * 365.25,
                "trend_significance": abs(slope) > 0.1,  # >0.1 mm/day is significant
                "acceleration_detected": self._detect_acceleration(
                    up_displacement, time_days
                ),
            }

        # Anomaly detection
        if len(up_displacement) >= 5:
            # Detect sudden changes
            anomalies = self._detect_displacement_anomalies(up_displacement)
            analysis["anomaly_detection"] = {
                "anomalous_periods": anomalies,
                "num_anomalies": len(anomalies),
                "largest_anomaly_mm": (
                    max([a["magnitude"] for a in anomalies]) if anomalies else 0
                ),
            }

        # Volcanic interpretation
        max_uplift = analysis["summary_statistics"]["max_uplift_mm"]
        trend_rate = analysis["trend_analysis"].get("linear_trend_mm_year", 0)

        if max_uplift > 50 and trend_rate > 10:
            interpretation = "significant_magma_intrusion"
            risk_level = "high"
        elif max_uplift > 20 or abs(trend_rate) > 5:
            interpretation = "moderate_volcanic_unrest"
            risk_level = "moderate"
        elif max_uplift > 10 or abs(trend_rate) > 2:
            interpretation = "minor_volcanic_activity"
            risk_level = "low"
        else:
            interpretation = "normal_background_activity"
            risk_level = "normal"

        analysis["volcanic_interpretation"] = {
            "interpretation": interpretation,
            "risk_level": risk_level,
            "confidence": 0.8,  # Would be calculated based on data quality
            "recommendations": self._generate_monitoring_recommendations(
                interpretation, risk_level
            ),
        }

        return analysis

    def _detect_acceleration(
        self, displacement: List[float], time_days: List[float]
    ) -> bool:
        """Detect acceleration in displacement time series"""

        if len(displacement) < 4:
            return False

        # Calculate second derivative (acceleration)
        velocities = np.diff(displacement) / np.diff(time_days)
        accelerations = np.diff(velocities) / np.diff(time_days[1:])

        # Check if acceleration exceeds threshold
        acceleration_threshold = 0.01  # mm/day²
        return np.any(np.abs(accelerations) > acceleration_threshold)

    def _detect_displacement_anomalies(
        self, displacement: List[float]
    ) -> List[Dict[str, Any]]:
        """Detect anomalous periods in displacement time series"""

        anomalies = []

        # Use simple threshold-based detection
        displacement_array = np.array(displacement)
        mean_disp = np.mean(displacement_array)
        std_disp = np.std(displacement_array)

        # Detect outliers (>2 standard deviations)
        outlier_threshold = 2.0
        outlier_indices = np.where(
            np.abs(displacement_array - mean_disp) > outlier_threshold * std_disp
        )[0]

        for idx in outlier_indices:
            anomalies.append(
                {
                    "index": int(idx),
                    "magnitude": float(displacement_array[idx] - mean_disp),
                    "severity": (
                        "high"
                        if abs(displacement_array[idx] - mean_disp) > 3 * std_disp
                        else "moderate"
                    ),
                }
            )

        return anomalies

    def _generate_monitoring_recommendations(
        self, interpretation: str, risk_level: str
    ) -> List[str]:
        """Generate monitoring recommendations based on deformation analysis"""

        recommendations = []

        if risk_level == "high":
            recommendations.extend(
                [
                    "Increase monitoring frequency to daily InSAR acquisitions",
                    "Deploy additional GPS stations around the volcano",
                    "Enhance seismic monitoring network",
                    "Conduct gas emission measurements",
                    "Prepare evacuation plans for nearby communities",
                    "Issue volcanic alert to relevant authorities",
                ]
            )
        elif risk_level == "moderate":
            recommendations.extend(
                [
                    "Continue regular InSAR monitoring",
                    "Increase seismic monitoring sensitivity",
                    "Monitor gas emissions weekly",
                    "Review evacuation procedures",
                    "Inform local authorities of increased activity",
                ]
            )
        elif risk_level == "low":
            recommendations.extend(
                [
                    "Maintain current monitoring schedule",
                    "Continue routine data analysis",
                    "Monitor for changes in deformation patterns",
                ]
            )
        else:  # normal
            recommendations.extend(
                ["Continue baseline monitoring", "Archive data for future reference"]
            )

        return recommendations

    def _generate_deformation_alerts(
        self, analysis: Dict[str, Any], volcano_name: str
    ) -> List[Dict[str, Any]]:
        """Generate alerts based on deformation analysis"""

        alerts = []

        if not analysis:
            return alerts

        volcanic_interp = analysis.get("volcanic_interpretation", {})
        risk_level = volcanic_interp.get("risk_level", "normal")
        interpretation = volcanic_interp.get(
            "interpretation", "normal_background_activity"
        )

        summary_stats = analysis.get("summary_statistics", {})
        max_uplift = summary_stats.get("max_uplift_mm", 0)

        if risk_level == "high":
            alerts.append(
                {
                    "alert_level": "HIGH",
                    "alert_type": "SIGNIFICANT_DEFORMATION",
                    "volcano_name": volcano_name,
                    "message": f"Significant ground deformation detected: {max_uplift:.1f}mm uplift. {interpretation}.",
                    "timestamp": datetime.utcnow().isoformat(),
                    "recommendations": volcanic_interp.get("recommendations", []),
                    "confidence": volcanic_interp.get("confidence", 0.8),
                }
            )
        elif risk_level == "moderate":
            alerts.append(
                {
                    "alert_level": "MODERATE",
                    "alert_type": "ELEVATED_DEFORMATION",
                    "volcano_name": volcano_name,
                    "message": f"Elevated ground deformation detected: {max_uplift:.1f}mm uplift. {interpretation}.",
                    "timestamp": datetime.utcnow().isoformat(),
                    "recommendations": volcanic_interp.get("recommendations", []),
                    "confidence": volcanic_interp.get("confidence", 0.8),
                }
            )

        # Check for anomalies
        anomaly_detection = analysis.get("anomaly_detection", {})
        if anomaly_detection.get("num_anomalies", 0) > 0:
            alerts.append(
                {
                    "alert_level": "INFO",
                    "alert_type": "DEFORMATION_ANOMALY",
                    "volcano_name": volcano_name,
                    "message": f'Detected {anomaly_detection["num_anomalies"]} anomalous deformation periods.',
                    "timestamp": datetime.utcnow().isoformat(),
                    "details": anomaly_detection,
                    "confidence": 0.7,
                }
            )

        return alerts

    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""

        return {
            "active_jobs": len(self.active_jobs),
            "completed_products": len(self.completed_products),
            "time_series_locations": len(self.time_series_data),
            "job_details": {
                job_id: {
                    "volcano_name": info["volcano_name"],
                    "status": info["status"],
                    "submitted_at": info["submitted_at"].isoformat(),
                }
                for job_id, info in self.active_jobs.items()
            },
        }
