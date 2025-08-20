"""
Advanced Satellite Acquisition Planning and ASF Integration

This module provides comprehensive satellite data acquisition planning and scheduling
capabilities with deep integration to Alaska Satellite Facility (ASF) APIs.

Key Features:
- Multi-satellite mission planning and optimization
- Automated data discovery and download workflows
- Real-time satellite orbit prediction and coverage analysis
- Intelligent acquisition scheduling with priority management
- Data quality assessment and validation
- Cost optimization for commercial satellite data
- Integration with ASF DAAC, ESA Copernicus, and JAXA G-Portal
- Automated processing pipeline triggers
- Advanced filtering and selection criteria
- Time series analysis for optimal temporal baselines

Scientific Applications:
- Volcanic deformation monitoring campaigns
- Earthquake cycle studies with optimal repeat intervals
- Long-term subsidence monitoring programs
- Emergency response rapid data acquisition
- Climate change impact assessments
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import asyncio
import aiohttp
import requests
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import warnings
from scipy.spatial.distance import cdist
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
import geopandas as gpd
import h5py
import zipfile
import tempfile
import shutil
import hashlib
import xml.etree.ElementTree as ET
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import schedule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()

class SatelliteMission(Enum):
    SENTINEL1 = "sentinel1"
    ALOS2 = "alos2"
    TERRASAR_X = "terrasar_x"
    COSMO_SKYMED = "cosmo_skymed"
    RADARSAT2 = "radarsat2"
    ICEYE = "iceye"
    CAPELLA = "capella"

class ProcessingLevel(Enum):
    RAW = "raw"
    SLC = "slc" 
    GRD = "grd"
    COH = "coh"
    GEOC = "geoc"

class AcquisitionPriority(Enum):
    EMERGENCY = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    ARCHIVE = 5

class DataProvider(Enum):
    ASF = "asf"
    ESA = "esa"
    JAXA = "jaxa"
    DLR = "dlr"
    ASI = "asi"
    CSA = "csa"
    COMMERCIAL = "commercial"

@dataclass
class GeographicBounds:
    north: float
    south: float
    east: float
    west: float
    
    def to_polygon(self) -> Polygon:
        return box(self.west, self.south, self.east, self.north)
    
    def to_wkt(self) -> str:
        return self.to_polygon().wkt

@dataclass
class SatelliteOrbit:
    mission: SatelliteMission
    satellite_id: str
    orbit_number: int
    pass_direction: str  # ASCENDING or DESCENDING
    look_direction: str  # LEFT or RIGHT
    start_time: datetime
    end_time: datetime
    footprint: Polygon
    coverage_area: float  # km²
    incidence_angle: float
    heading_angle: float

@dataclass
class AcquisitionPlan:
    plan_id: str
    name: str
    description: str
    target_area: GeographicBounds
    priority: AcquisitionPriority
    start_date: datetime
    end_date: datetime
    missions: List[SatelliteMission]
    processing_levels: List[ProcessingLevel]
    temporal_baseline_max: int = 24  # days
    perpendicular_baseline_max: int = 200  # meters
    coherence_threshold: float = 0.3
    cloud_cover_max: float = 20.0  # percent
    min_overlap: float = 0.8  # minimum scene overlap
    auto_download: bool = True
    auto_process: bool = False
    created_date: datetime = field(default_factory=datetime.now)

@dataclass 
class SatelliteProduct:
    product_id: str
    mission: SatelliteMission
    provider: DataProvider
    processing_level: ProcessingLevel
    acquisition_date: datetime
    footprint: Polygon
    orbit_number: int
    pass_direction: str
    polarization: List[str]
    incidence_angle: float
    file_size: int  # bytes
    download_url: str
    metadata_url: Optional[str] = None
    preview_url: Optional[str] = None
    cloud_cover: Optional[float] = None
    coherence_estimate: Optional[float] = None
    baseline_info: Optional[Dict] = None
    quality_flags: Dict[str, bool] = field(default_factory=dict)
    cost: Optional[float] = None  # USD
    availability_date: Optional[datetime] = None

class SatelliteProductDB(Base):
    """Database model for satellite products"""
    __tablename__ = 'satellite_products'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(String(100), unique=True, nullable=False)
    mission = Column(String(20), nullable=False)
    provider = Column(String(20), nullable=False)
    processing_level = Column(String(10), nullable=False)
    acquisition_date = Column(DateTime, nullable=False)
    footprint_wkt = Column(Text, nullable=False)
    orbit_number = Column(Integer, nullable=False)
    pass_direction = Column(String(15), nullable=False)
    polarization = Column(String(50), nullable=False)
    incidence_angle = Column(Float, nullable=False)
    file_size = Column(Integer, nullable=False)
    download_url = Column(Text, nullable=False)
    metadata_url = Column(Text)
    preview_url = Column(Text)
    cloud_cover = Column(Float)
    coherence_estimate = Column(Float)
    quality_score = Column(Float)
    cost = Column(Float)
    availability_date = Column(DateTime)
    downloaded = Column(Boolean, default=False)
    processed = Column(Boolean, default=False)
    created_date = Column(DateTime, default=datetime.utcnow)

class ASFDataClient:
    """Advanced Alaska Satellite Facility data client"""
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.base_url = "https://api.daac.asf.alaska.edu"
        self.search_url = f"{self.base_url}/services/search/param"
        self.download_url = f"{self.base_url}/services/utils/files_to_metalinks"
        
        self.session = aiohttp.ClientSession()
        self.auth_token = None
        self.request_count = 0
        self.last_request_time = 0
        
        # Rate limiting
        self.max_requests_per_second = 10
        self.request_delay = 1.0 / self.max_requests_per_second
    
    async def authenticate(self) -> bool:
        """Authenticate with ASF API"""
        try:
            auth_url = f"{self.base_url}/auth/login"
            auth_data = {
                "username": self.username,
                "password": self.password
            }
            
            async with self.session.post(auth_url, json=auth_data) as response:
                if response.status == 200:
                    result = await response.json()
                    self.auth_token = result.get('access_token')
                    logger.info("Successfully authenticated with ASF")
                    return True
                else:
                    logger.error(f"ASF authentication failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"ASF authentication error: {e}")
            return False
    
    async def search_products(self, 
                            bounds: GeographicBounds,
                            start_date: datetime,
                            end_date: datetime,
                            missions: List[SatelliteMission] = None,
                            processing_levels: List[ProcessingLevel] = None,
                            max_results: int = 1000) -> List[SatelliteProduct]:
        """Search for satellite products"""
        
        await self._rate_limit()
        
        # Build search parameters
        params = {
            "intersectsWith": bounds.to_wkt(),
            "start": start_date.isoformat() + "Z",
            "end": end_date.isoformat() + "Z",
            "output": "JSON",
            "maxResults": max_results
        }
        
        # Add mission filters
        if missions:
            platform_list = []
            for mission in missions:
                if mission == SatelliteMission.SENTINEL1:
                    platform_list.extend(["SENTINEL-1A", "SENTINEL-1B"])
                elif mission == SatelliteMission.ALOS2:
                    platform_list.append("ALOS-2")
                elif mission == SatelliteMission.TERRASAR_X:
                    platform_list.extend(["TERRASAR-X", "TANDEM-X"])
            
            if platform_list:
                params["platform"] = ",".join(platform_list)
        
        # Add processing level filters
        if processing_levels:
            level_list = []
            for level in processing_levels:
                if level == ProcessingLevel.SLC:
                    level_list.append("SLC")
                elif level == ProcessingLevel.GRD:
                    level_list.append("GRD_HD")
            
            if level_list:
                params["processingLevel"] = ",".join(level_list)
        
        try:
            logger.info(f"Searching ASF with parameters: {params}")
            
            async with self.session.get(self.search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    products = []
                    for item in data[0]:  # ASF returns list with single element containing results
                        try:
                            product = self._parse_asf_product(item)
                            if product:
                                products.append(product)
                        except Exception as e:
                            logger.warning(f"Failed to parse ASF product: {e}")
                            continue
                    
                    logger.info(f"Found {len(products)} products from ASF")
                    return products
                else:
                    logger.error(f"ASF search failed: {response.status} - {await response.text()}")
                    return []
                    
        except Exception as e:
            logger.error(f"ASF search error: {e}")
            return []
    
    def _parse_asf_product(self, item: Dict) -> Optional[SatelliteProduct]:
        """Parse ASF API response item into SatelliteProduct"""
        
        try:
            # Extract mission
            platform = item.get('platform', '').upper()
            if 'SENTINEL-1' in platform:
                mission = SatelliteMission.SENTINEL1
            elif 'ALOS-2' in platform:
                mission = SatelliteMission.ALOS2
            elif 'TERRASAR-X' in platform or 'TANDEM-X' in platform:
                mission = SatelliteMission.TERRASAR_X
            else:
                return None
            
            # Extract processing level
            proc_level = item.get('processingLevel', '').upper()
            if proc_level == 'SLC':
                processing_level = ProcessingLevel.SLC
            elif proc_level.startswith('GRD'):
                processing_level = ProcessingLevel.GRD
            else:
                processing_level = ProcessingLevel.RAW
            
            # Parse geometry
            footprint_wkt = item.get('wkt', '')
            if footprint_wkt:
                from shapely import wkt
                footprint = wkt.loads(footprint_wkt)
            else:
                # Fallback to bounding box
                bbox = item.get('bbox', [])
                if len(bbox) == 4:
                    footprint = box(bbox[0], bbox[1], bbox[2], bbox[3])
                else:
                    return None
            
            # Parse acquisition date
            start_time_str = item.get('startTime', '')
            if start_time_str:
                acquisition_date = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            else:
                return None
            
            # Extract other fields
            product_id = item.get('granuleName', item.get('sceneName', ''))
            orbit_number = int(item.get('orbit', 0))
            pass_direction = item.get('flightDirection', 'UNKNOWN').upper()
            polarization = item.get('polarization', '').split(',')
            incidence_angle = float(item.get('centerLat', 0))  # Approximation
            file_size = int(item.get('bytes', 0))
            
            # Download URLs
            download_url = item.get('url', '')
            
            product = SatelliteProduct(
                product_id=product_id,
                mission=mission,
                provider=DataProvider.ASF,
                processing_level=processing_level,
                acquisition_date=acquisition_date,
                footprint=footprint,
                orbit_number=orbit_number,
                pass_direction=pass_direction,
                polarization=polarization,
                incidence_angle=incidence_angle,
                file_size=file_size,
                download_url=download_url,
                metadata_url=item.get('metalink', ''),
                preview_url=item.get('browse', '')
            )
            
            return product
            
        except Exception as e:
            logger.error(f"Error parsing ASF product: {e}")
            return None
    
    async def download_product(self, 
                             product: SatelliteProduct, 
                             output_path: str,
                             chunk_size: int = 8192) -> bool:
        """Download satellite product"""
        
        try:
            logger.info(f"Downloading {product.product_id}")
            
            # Create output directory
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            async with self.session.get(product.download_url) as response:
                if response.status == 200:
                    total_size = int(response.headers.get('Content-Length', 0))
                    downloaded_size = 0
                    
                    with open(output_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # Progress reporting
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                if downloaded_size % (chunk_size * 100) == 0:  # Report every 100 chunks
                                    logger.info(f"Download progress: {progress:.1f}%")
                    
                    logger.info(f"Successfully downloaded {product.product_id}")
                    return True
                else:
                    logger.error(f"Download failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    async def close(self):
        """Close the HTTP session"""
        await self.session.close()

class SatelliteScheduler:
    """Advanced satellite acquisition scheduling system"""
    
    def __init__(self, database_url: str = "sqlite:///satellite_scheduler.db"):
        # Database setup
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Data clients
        self.asf_client = None
        self.data_providers = {}
        
        # Scheduling parameters
        self.scheduling_horizon = 365  # days
        self.optimization_weights = {
            'temporal_baseline': 0.3,
            'perpendicular_baseline': 0.2,
            'coherence': 0.2,
            'coverage': 0.15,
            'cost': 0.15
        }
        
        # Active plans
        self.acquisition_plans = {}
        self.active_downloads = {}
        
        # Statistics
        self.stats = {
            'products_discovered': 0,
            'products_downloaded': 0,
            'data_volume': 0,  # GB
            'cost_total': 0.0,  # USD
            'processing_queue': 0
        }
    
    def initialize_data_providers(self, credentials: Dict[str, Dict]):
        """Initialize data provider clients"""
        
        # ASF initialization
        if 'asf' in credentials:
            asf_creds = credentials['asf']
            self.asf_client = ASFDataClient(
                asf_creds['username'], 
                asf_creds['password']
            )
        
        logger.info("Data providers initialized")
    
    async def create_acquisition_plan(self, plan: AcquisitionPlan) -> str:
        """Create new acquisition plan"""
        
        logger.info(f"Creating acquisition plan: {plan.name}")
        
        # Store plan
        self.acquisition_plans[plan.plan_id] = plan
        
        # Schedule initial search
        await self._schedule_plan_execution(plan)
        
        return plan.plan_id
    
    async def _schedule_plan_execution(self, plan: AcquisitionPlan):
        """Schedule execution of acquisition plan"""
        
        # Immediate execution for testing
        await self.execute_acquisition_plan(plan.plan_id)
        
        # In production, this would use a proper scheduler like Celery
        # schedule.every().day.at("06:00").do(self.execute_acquisition_plan, plan.plan_id)
    
    async def execute_acquisition_plan(self, plan_id: str) -> Dict:
        """Execute acquisition plan"""
        
        if plan_id not in self.acquisition_plans:
            raise ValueError(f"Plan {plan_id} not found")
        
        plan = self.acquisition_plans[plan_id]
        logger.info(f"Executing acquisition plan: {plan.name}")
        
        results = {
            'plan_id': plan_id,
            'execution_time': datetime.now(),
            'products_found': 0,
            'products_scheduled': 0,
            'estimated_cost': 0.0,
            'coverage_analysis': {}
        }
        
        try:
            # Search for products
            all_products = []
            
            for mission in plan.missions:
                if mission == SatelliteMission.SENTINEL1 and self.asf_client:
                    products = await self.asf_client.search_products(
                        bounds=plan.target_area,
                        start_date=plan.start_date,
                        end_date=plan.end_date,
                        missions=[mission],
                        processing_levels=plan.processing_levels
                    )
                    all_products.extend(products)
            
            results['products_found'] = len(all_products)
            logger.info(f"Found {len(all_products)} products for plan {plan.name}")
            
            if all_products:
                # Optimize product selection
                selected_products = await self._optimize_product_selection(all_products, plan)
                results['products_scheduled'] = len(selected_products)
                
                # Store products in database
                for product in selected_products:
                    await self._store_product(product)
                
                # Schedule downloads
                if plan.auto_download:
                    await self._schedule_downloads(selected_products)
                
                # Coverage analysis
                results['coverage_analysis'] = self._analyze_coverage(selected_products, plan.target_area)
                
                # Cost estimation
                results['estimated_cost'] = sum(p.cost or 0 for p in selected_products)
                
                # Update statistics
                self.stats['products_discovered'] += len(all_products)
                
        except Exception as e:
            logger.error(f"Error executing plan {plan_id}: {e}")
            results['error'] = str(e)
        
        return results
    
    async def _optimize_product_selection(self, 
                                        products: List[SatelliteProduct], 
                                        plan: AcquisitionPlan) -> List[SatelliteProduct]:
        """Optimize product selection based on plan criteria"""
        
        logger.info(f"Optimizing selection from {len(products)} products")
        
        # Filter by basic criteria
        filtered_products = []
        
        for product in products:
            # Check overlap with target area
            overlap = self._calculate_overlap(product.footprint, plan.target_area.to_polygon())
            if overlap < plan.min_overlap:
                continue
            
            # Check cloud cover if available
            if product.cloud_cover and product.cloud_cover > plan.cloud_cover_max:
                continue
            
            filtered_products.append(product)
        
        if not filtered_products:
            return []
        
        # Advanced optimization
        if len(filtered_products) > 50:
            # Use optimization algorithm for large sets
            optimized_products = await self._genetic_algorithm_optimization(filtered_products, plan)
        else:
            # Use heuristic selection for small sets
            optimized_products = await self._heuristic_selection(filtered_products, plan)
        
        logger.info(f"Selected {len(optimized_products)} optimal products")
        return optimized_products
    
    async def _genetic_algorithm_optimization(self,
                                            products: List[SatelliteProduct],
                                            plan: AcquisitionPlan) -> List[SatelliteProduct]:
        """Use genetic algorithm for product selection optimization"""
        
        # Create optimization problem
        def objective_function(selection_binary):
            selected_products = [products[i] for i in range(len(products)) if selection_binary[i] > 0.5]
            
            if not selected_products:
                return 1e6  # High penalty for empty selection
            
            # Calculate multiple objectives
            score = 0.0
            
            # Temporal baseline optimization
            temporal_score = self._evaluate_temporal_distribution(selected_products)
            score += self.optimization_weights['temporal_baseline'] * temporal_score
            
            # Coverage optimization
            coverage_score = self._evaluate_coverage_score(selected_products, plan.target_area)
            score += self.optimization_weights['coverage'] * coverage_score
            
            # Cost optimization (minimize)
            total_cost = sum(p.cost or 0 for p in selected_products)
            cost_score = 1.0 / (1.0 + total_cost / 1000)  # Normalize cost
            score += self.optimization_weights['cost'] * cost_score
            
            # Coherence optimization (if available)
            coherence_score = np.mean([p.coherence_estimate or 0.5 for p in selected_products])
            score += self.optimization_weights['coherence'] * coherence_score
            
            return -score  # Minimize negative score (maximize original score)
        
        # Set bounds
        bounds = [(0, 1) for _ in range(len(products))]
        
        # Run optimization
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=100,
            popsize=15,
            seed=42
        )
        
        # Extract selected products
        selection_binary = result.x
        selected_indices = [i for i in range(len(products)) if selection_binary[i] > 0.5]
        selected_products = [products[i] for i in selected_indices]
        
        return selected_products
    
    async def _heuristic_selection(self,
                                 products: List[SatelliteProduct],
                                 plan: AcquisitionPlan) -> List[SatelliteProduct]:
        """Heuristic-based product selection"""
        
        # Sort products by acquisition date
        products_sorted = sorted(products, key=lambda p: p.acquisition_date)
        
        selected_products = []
        last_acquisition_date = None
        
        for product in products_sorted:
            # Check temporal baseline constraint
            if last_acquisition_date:
                temporal_gap = (product.acquisition_date - last_acquisition_date).days
                if temporal_gap < 6:  # Minimum 6 days between acquisitions
                    continue
                if temporal_gap > plan.temporal_baseline_max:
                    continue
            
            # Quality checks
            quality_score = self._calculate_quality_score(product)
            if quality_score < 0.6:  # Minimum quality threshold
                continue
            
            selected_products.append(product)
            last_acquisition_date = product.acquisition_date
            
            # Limit selection size
            if len(selected_products) >= 20:
                break
        
        return selected_products
    
    def _calculate_overlap(self, footprint1: Polygon, footprint2: Polygon) -> float:
        """Calculate overlap ratio between two polygons"""
        try:
            intersection = footprint1.intersection(footprint2)
            return intersection.area / footprint1.area
        except:
            return 0.0
    
    def _evaluate_temporal_distribution(self, products: List[SatelliteProduct]) -> float:
        """Evaluate temporal distribution of products"""
        
        if len(products) < 2:
            return 0.5
        
        # Calculate temporal gaps
        dates = sorted([p.acquisition_date for p in products])
        gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        
        # Prefer regular temporal sampling
        mean_gap = np.mean(gaps)
        gap_variance = np.var(gaps)
        
        # Score based on regularity and reasonable gaps
        regularity_score = 1.0 / (1.0 + gap_variance / max(mean_gap, 1))
        gap_score = 1.0 / (1.0 + abs(mean_gap - 12) / 12)  # Prefer ~12 day gaps
        
        return 0.6 * regularity_score + 0.4 * gap_score
    
    def _evaluate_coverage_score(self, products: List[SatelliteProduct], target_area: GeographicBounds) -> float:
        """Evaluate coverage score for selected products"""
        
        if not products:
            return 0.0
        
        # Calculate union of all footprints
        footprints = [p.footprint for p in products]
        try:
            union_footprint = unary_union(footprints)
            target_polygon = target_area.to_polygon()
            
            coverage_ratio = union_footprint.intersection(target_polygon).area / target_polygon.area
            return min(coverage_ratio, 1.0)
        except:
            return 0.5
    
    def _calculate_quality_score(self, product: SatelliteProduct) -> float:
        """Calculate overall quality score for product"""
        
        score = 0.7  # Base score
        
        # Coherence bonus
        if product.coherence_estimate:
            score += 0.2 * product.coherence_estimate
        
        # Cloud cover penalty
        if product.cloud_cover:
            score -= 0.3 * (product.cloud_cover / 100)
        
        # Incidence angle preference (around 35-40 degrees is optimal)
        if product.incidence_angle:
            angle_score = 1.0 - abs(product.incidence_angle - 37.5) / 37.5
            score += 0.1 * angle_score
        
        return max(0.0, min(1.0, score))
    
    async def _store_product(self, product: SatelliteProduct):
        """Store product information in database"""
        
        try:
            # Check if product already exists
            existing = self.session.query(SatelliteProductDB).filter_by(
                product_id=product.product_id
            ).first()
            
            if existing:
                return
            
            # Create new database record
            db_product = SatelliteProductDB(
                product_id=product.product_id,
                mission=product.mission.value,
                provider=product.provider.value,
                processing_level=product.processing_level.value,
                acquisition_date=product.acquisition_date,
                footprint_wkt=product.footprint.wkt,
                orbit_number=product.orbit_number,
                pass_direction=product.pass_direction,
                polarization=','.join(product.polarization),
                incidence_angle=product.incidence_angle,
                file_size=product.file_size,
                download_url=product.download_url,
                metadata_url=product.metadata_url,
                preview_url=product.preview_url,
                cloud_cover=product.cloud_cover,
                coherence_estimate=product.coherence_estimate,
                quality_score=self._calculate_quality_score(product),
                cost=product.cost
            )
            
            self.session.add(db_product)
            self.session.commit()
            
        except Exception as e:
            logger.error(f"Error storing product {product.product_id}: {e}")
            self.session.rollback()
    
    async def _schedule_downloads(self, products: List[SatelliteProduct]):
        """Schedule product downloads"""
        
        for product in products:
            download_id = f"download_{product.product_id}_{int(time.time())}"
            self.active_downloads[download_id] = {
                'product': product,
                'status': 'scheduled',
                'created_time': datetime.now()
            }
        
        # Start downloads asynchronously
        asyncio.create_task(self._process_download_queue())
    
    async def _process_download_queue(self):
        """Process download queue"""
        
        for download_id, download_info in self.active_downloads.items():
            if download_info['status'] == 'scheduled':
                download_info['status'] = 'downloading'
                
                try:
                    product = download_info['product']
                    output_path = f"data/{product.mission.value}/{product.product_id}.zip"
                    
                    success = await self.asf_client.download_product(product, output_path)
                    
                    if success:
                        download_info['status'] = 'completed'
                        download_info['completion_time'] = datetime.now()
                        
                        # Update database
                        db_product = self.session.query(SatelliteProductDB).filter_by(
                            product_id=product.product_id
                        ).first()
                        if db_product:
                            db_product.downloaded = True
                            self.session.commit()
                        
                        # Update statistics
                        self.stats['products_downloaded'] += 1
                        self.stats['data_volume'] += product.file_size / (1024**3)  # GB
                        
                        logger.info(f"Download completed: {product.product_id}")
                    else:
                        download_info['status'] = 'failed'
                        logger.error(f"Download failed: {product.product_id}")
                
                except Exception as e:
                    download_info['status'] = 'error'
                    download_info['error'] = str(e)
                    logger.error(f"Download error for {product.product_id}: {e}")
    
    def _analyze_coverage(self, products: List[SatelliteProduct], target_area: GeographicBounds) -> Dict:
        """Analyze spatial and temporal coverage"""
        
        analysis = {
            'spatial_coverage': 0.0,
            'temporal_span_days': 0,
            'acquisition_count': len(products),
            'orbit_diversity': {},
            'temporal_gaps': []
        }
        
        if not products:
            return analysis
        
        # Spatial coverage analysis
        footprints = [p.footprint for p in products]
        union_footprint = unary_union(footprints)
        target_polygon = target_area.to_polygon()
        
        analysis['spatial_coverage'] = union_footprint.intersection(target_polygon).area / target_polygon.area
        
        # Temporal analysis
        dates = sorted([p.acquisition_date for p in products])
        if len(dates) > 1:
            analysis['temporal_span_days'] = (dates[-1] - dates[0]).days
            
            # Calculate temporal gaps
            gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            analysis['temporal_gaps'] = gaps
        
        # Orbit diversity
        orbits = {}
        for product in products:
            orbit_key = f"{product.pass_direction}_{product.orbit_number % 100}"
            orbits[orbit_key] = orbits.get(orbit_key, 0) + 1
        
        analysis['orbit_diversity'] = orbits
        
        return analysis
    
    async def get_plan_status(self, plan_id: str) -> Dict:
        """Get status of acquisition plan"""
        
        if plan_id not in self.acquisition_plans:
            raise ValueError(f"Plan {plan_id} not found")
        
        plan = self.acquisition_plans[plan_id]
        
        # Query database for products
        db_products = self.session.query(SatelliteProductDB).filter(
            SatelliteProductDB.acquisition_date >= plan.start_date,
            SatelliteProductDB.acquisition_date <= plan.end_date
        ).all()
        
        # Count downloads and processing
        downloaded_count = sum(1 for p in db_products if p.downloaded)
        processed_count = sum(1 for p in db_products if p.processed)
        
        status = {
            'plan_id': plan_id,
            'plan_name': plan.name,
            'status': 'active',
            'products_found': len(db_products),
            'products_downloaded': downloaded_count,
            'products_processed': processed_count,
            'data_volume_gb': sum(p.file_size for p in db_products) / (1024**3),
            'estimated_cost': sum(p.cost or 0 for p in db_products),
            'coverage_analysis': {},
            'last_update': datetime.now()
        }
        
        return status
    
    async def get_orbit_predictions(self, 
                                  mission: SatelliteMission,
                                  target_area: GeographicBounds,
                                  start_date: datetime,
                                  days_ahead: int = 30) -> List[SatelliteOrbit]:
        """Get predicted satellite orbits for planning"""
        
        # This would interface with orbital prediction services
        # For now, return mock data
        
        predictions = []
        current_date = start_date
        
        for day in range(days_ahead):
            current_date = start_date + timedelta(days=day)
            
            # Mock orbit prediction
            if mission == SatelliteMission.SENTINEL1:
                # Sentinel-1 has 12-day repeat cycle
                if day % 12 < 6:  # Simulate coverage every 12 days
                    for satellite_id in ['S1A', 'S1B']:
                        orbit = SatelliteOrbit(
                            mission=mission,
                            satellite_id=satellite_id,
                            orbit_number=day * 100 + hash(satellite_id) % 100,
                            pass_direction='DESCENDING',
                            look_direction='RIGHT',
                            start_time=current_date.replace(hour=6, minute=0),
                            end_time=current_date.replace(hour=6, minute=10),
                            footprint=target_area.to_polygon(),
                            coverage_area=target_area.to_polygon().area / 1e6,  # km²
                            incidence_angle=37.5,
                            heading_angle=192.0
                        )
                        predictions.append(orbit)
        
        return predictions
    
    def generate_acquisition_report(self, plan_id: str) -> Dict:
        """Generate comprehensive acquisition report"""
        
        if plan_id not in self.acquisition_plans:
            raise ValueError(f"Plan {plan_id} not found")
        
        plan = self.acquisition_plans[plan_id]
        
        # Query database
        db_products = self.session.query(SatelliteProductDB).filter(
            SatelliteProductDB.acquisition_date >= plan.start_date,
            SatelliteProductDB.acquisition_date <= plan.end_date
        ).all()
        
        # Generate statistics
        total_products = len(db_products)
        downloaded_products = sum(1 for p in db_products if p.downloaded)
        processed_products = sum(1 for p in db_products if p.processed)
        total_data_volume = sum(p.file_size for p in db_products) / (1024**3)  # GB
        total_cost = sum(p.cost or 0 for p in db_products)
        
        # Mission breakdown
        mission_stats = {}
        for product in db_products:
            mission = product.mission
            if mission not in mission_stats:
                mission_stats[mission] = {'count': 0, 'data_volume': 0, 'cost': 0}
            
            mission_stats[mission]['count'] += 1
            mission_stats[mission]['data_volume'] += product.file_size / (1024**3)
            mission_stats[mission]['cost'] += product.cost or 0
        
        # Temporal analysis
        if db_products:
            acquisition_dates = [p.acquisition_date for p in db_products]
            temporal_span = (max(acquisition_dates) - min(acquisition_dates)).days
            avg_temporal_gap = temporal_span / max(len(acquisition_dates) - 1, 1)
        else:
            temporal_span = 0
            avg_temporal_gap = 0
        
        report = {
            'plan_information': {
                'plan_id': plan_id,
                'name': plan.name,
                'description': plan.description,
                'target_area': {
                    'north': plan.target_area.north,
                    'south': plan.target_area.south,
                    'east': plan.target_area.east,
                    'west': plan.target_area.west
                },
                'time_range': {
                    'start': plan.start_date.isoformat(),
                    'end': plan.end_date.isoformat()
                },
                'priority': plan.priority.name
            },
            'acquisition_summary': {
                'total_products': total_products,
                'downloaded_products': downloaded_products,
                'processed_products': processed_products,
                'download_completion_rate': downloaded_products / max(total_products, 1) * 100,
                'processing_completion_rate': processed_products / max(total_products, 1) * 100,
                'total_data_volume_gb': total_data_volume,
                'total_cost_usd': total_cost
            },
            'mission_breakdown': mission_stats,
            'temporal_analysis': {
                'temporal_span_days': temporal_span,
                'average_temporal_gap_days': avg_temporal_gap,
                'acquisition_frequency': len(db_products) / max(temporal_span / 365.25, 1)  # per year
            },
            'quality_metrics': {
                'average_quality_score': np.mean([p.quality_score or 0.5 for p in db_products]),
                'high_quality_products': sum(1 for p in db_products if (p.quality_score or 0) > 0.8),
                'low_quality_products': sum(1 for p in db_products if (p.quality_score or 0) < 0.4)
            },
            'generated_date': datetime.now().isoformat()
        }
        
        return report
    
    def visualize_coverage(self, plan_id: str, output_path: str = None):
        """Visualize spatial and temporal coverage"""
        
        if plan_id not in self.acquisition_plans:
            raise ValueError(f"Plan {plan_id} not found")
        
        plan = self.acquisition_plans[plan_id]
        
        # Query products
        db_products = self.session.query(SatelliteProductDB).filter(
            SatelliteProductDB.acquisition_date >= plan.start_date,
            SatelliteProductDB.acquisition_date <= plan.end_date
        ).all()
        
        if not db_products:
            logger.warning(f"No products found for plan {plan_id}")
            return
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Spatial coverage map
        ax1.set_title(f'Spatial Coverage - {plan.name}')
        
        # Plot target area
        target_bounds = plan.target_area
        ax1.add_patch(plt.Rectangle(
            (target_bounds.west, target_bounds.south),
            target_bounds.east - target_bounds.west,
            target_bounds.north - target_bounds.south,
            fill=False, edgecolor='red', linewidth=2, label='Target Area'
        ))
        
        # Plot product footprints
        for i, product in enumerate(db_products[:20]):  # Limit to first 20 for clarity
            try:
                footprint_wkt = product.footprint_wkt
                from shapely import wkt
                footprint = wkt.loads(footprint_wkt)
                
                if hasattr(footprint, 'exterior'):
                    coords = footprint.exterior.coords
                    lons, lats = zip(*coords)
                    ax1.plot(lons, lats, alpha=0.3, linewidth=0.5)
            except:
                continue
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Temporal coverage timeline
        ax2.set_title(f'Temporal Coverage - {plan.name}')
        
        # Plot acquisition dates
        dates = [p.acquisition_date for p in db_products]
        missions = [p.mission for p in db_products]
        
        # Color code by mission
        mission_colors = {
            'sentinel1': 'blue',
            'alos2': 'green', 
            'terrasar_x': 'orange'
        }
        
        for i, (date, mission) in enumerate(zip(dates, missions)):
            color = mission_colors.get(mission, 'gray')
            ax2.scatter(date, i, c=color, s=20, alpha=0.7)
        
        ax2.set_xlabel('Acquisition Date')
        ax2.set_ylabel('Product Index')
        ax2.grid(True, alpha=0.3)
        
        # Add mission legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=8, label=mission.title())
                          for mission, color in mission_colors.items()]
        ax2.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Coverage visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    async def cleanup(self):
        """Cleanup resources"""
        
        if self.asf_client:
            await self.asf_client.close()
        
        self.session.close()
        logger.info("Satellite scheduler cleanup completed")

def create_sample_acquisition_plan() -> AcquisitionPlan:
    """Create sample acquisition plan for testing"""
    
    return AcquisitionPlan(
        plan_id="volcanic_monitoring_001",
        name="Kilauea Volcano Monitoring",
        description="Long-term deformation monitoring of Kilauea volcano using multi-satellite InSAR",
        target_area=GeographicBounds(
            north=19.6,
            south=19.3,
            east=-154.8,
            west=-155.4
        ),
        priority=AcquisitionPriority.HIGH,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 12, 31),
        missions=[SatelliteMission.SENTINEL1, SatelliteMission.ALOS2],
        processing_levels=[ProcessingLevel.SLC],
        temporal_baseline_max=24,
        perpendicular_baseline_max=200,
        coherence_threshold=0.3,
        min_overlap=0.8,
        auto_download=True,
        auto_process=False
    )

async def main():
    """Example usage of satellite scheduler"""
    
    # Initialize scheduler
    scheduler = SatelliteScheduler()
    
    # Initialize data providers (would use real credentials)
    credentials = {
        'asf': {
            'username': 'test_user',
            'password': 'test_password'
        }
    }
    scheduler.initialize_data_providers(credentials)
    
    # Create sample acquisition plan
    plan = create_sample_acquisition_plan()
    
    # Create and execute plan
    plan_id = await scheduler.create_acquisition_plan(plan)
    logger.info(f"Created acquisition plan: {plan_id}")
    
    # Get plan status
    status = await scheduler.get_plan_status(plan_id)
    logger.info(f"Plan status: {status}")
    
    # Generate report
    report = scheduler.generate_acquisition_report(plan_id)
    logger.info(f"Generated report: {json.dumps(report, indent=2, default=str)}")
    
    # Visualize coverage (would create plot)
    try:
        scheduler.visualize_coverage(plan_id, "coverage_analysis.png")
    except Exception as e:
        logger.warning(f"Could not create visualization: {e}")
    
    # Cleanup
    await scheduler.cleanup()

if __name__ == "__main__":
    asyncio.run(main())