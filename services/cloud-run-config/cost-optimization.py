#!/usr/bin/env python3
"""
Cloud Run Cost Optimization Engine

This module provides intelligent cost optimization for Cloud Run services using:
- Preemptible instances and spot pricing
- Dynamic resource allocation
- Usage pattern analysis
- Multi-region cost arbitrage
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

import aiohttp
from google.cloud import run_v2
from google.cloud import monitoring_v3
from google.cloud import billing_v1


class CostOptimizationStrategy(Enum):
    """Cost optimization strategies"""
    AGGRESSIVE = "aggressive"  # Maximum cost savings, may impact performance
    BALANCED = "balanced"  # Balance cost and performance
    CONSERVATIVE = "conservative"  # Minimal impact on performance
    CUSTOM = "custom"  # User-defined optimization rules


@dataclass
class CostMetrics:
    """Cost metrics for a service"""
    service_name: str
    region: str
    hourly_cost: float
    monthly_projection: float
    cpu_cost: float
    memory_cost: float
    request_cost: float
    instance_hours: float
    preemptible_savings: float
    timestamp: datetime


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation"""
    service_name: str
    region: str
    current_cost: float
    optimized_cost: float
    savings_percentage: float
    recommendations: List[str]
    risk_level: str  # low, medium, high
    implementation_priority: int  # 1-5, 1 being highest


@dataclass
class ResourceOptimization:
    """Resource optimization configuration"""
    cpu_limit: str
    memory_limit: str
    min_instances: int
    max_instances: int
    concurrency: int
    use_preemptible: bool
    scale_to_zero: bool
    timeout_seconds: int


class CloudRunCostOptimizer:
    """
    Intelligent cost optimization for Cloud Run services with:
    - Real-time cost monitoring and analysis
    - Preemptible instance management
    - Resource right-sizing recommendations
    - Multi-region cost arbitrage
    - Usage pattern-based optimization
    """
    
    def __init__(self, project_id: str, billing_account_id: str):
        self.project_id = project_id
        self.billing_account_id = billing_account_id
        self.run_client = run_v2.ServicesClient()
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.billing_client = billing_v1.CloudBillingClient()
        self.logger = logging.getLogger(__name__)
        
        # Regional cost factors (relative to us-central1)
        self.regional_cost_factors = {
            "us-central1": 1.0,      # Base pricing
            "us-east1": 0.95,        # Slightly cheaper
            "us-west1": 1.05,        # Slightly more expensive
            "europe-west1": 1.08,    # European pricing
            "europe-west4": 1.12,    # Premium European region
            "asia-northeast1": 1.15, # Japan pricing
            "asia-southeast1": 1.10, # Singapore pricing
            "australia-southeast1": 1.20  # Australia pricing
        }
        
        # Service-specific optimization profiles
        self.service_profiles = {
            "helios-core": {
                "cpu_intensive": True,
                "memory_intensive": True,
                "latency_sensitive": True,
                "batch_workload": False,
                "preemptible_tolerance": 0.3  # 30% can be preemptible
            },
            "data-ingestion": {
                "cpu_intensive": False,
                "memory_intensive": True,
                "latency_sensitive": False,
                "batch_workload": True,
                "preemptible_tolerance": 0.8  # 80% can be preemptible
            },
            "volcano-watchtower": {
                "cpu_intensive": True,
                "memory_intensive": False,
                "latency_sensitive": True,
                "batch_workload": False,
                "preemptible_tolerance": 0.2  # 20% can be preemptible
            },
            "api-gateway": {
                "cpu_intensive": False,
                "memory_intensive": False,
                "latency_sensitive": True,
                "batch_workload": False,
                "preemptible_tolerance": 0.1  # 10% can be preemptible
            }
        }
        
        # Current Cloud Run pricing (simplified, per hour)
        self.pricing = {
            "cpu_per_vcpu_hour": 0.00002400,  # $0.000024 per vCPU-hour
            "memory_per_gb_hour": 0.00000250,  # $0.0000025 per GB-hour
            "request_per_million": 0.40,       # $0.40 per million requests
            "preemptible_discount": 0.30       # 30% discount for preemptible
        }
    
    async def analyze_service_costs(self, service_name: str, region: str, days: int = 7) -> CostMetrics:
        """Analyze costs for a specific service over a time period"""
        try:
            # Get usage metrics from Cloud Monitoring
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            usage_metrics = await self._get_usage_metrics(service_name, region, start_time, end_time)
            
            # Calculate costs
            cpu_hours = usage_metrics.get("cpu_hours", 0)
            memory_gb_hours = usage_metrics.get("memory_gb_hours", 0)
            total_requests = usage_metrics.get("total_requests", 0)
            instance_hours = usage_metrics.get("instance_hours", 0)
            preemptible_hours = usage_metrics.get("preemptible_hours", 0)
            
            # Apply regional cost factor
            cost_factor = self.regional_cost_factors.get(region, 1.0)
            
            # Calculate component costs
            cpu_cost = cpu_hours * self.pricing["cpu_per_vcpu_hour"] * cost_factor
            memory_cost = memory_gb_hours * self.pricing["memory_per_gb_hour"] * cost_factor
            request_cost = (total_requests / 1_000_000) * self.pricing["request_per_million"]
            
            # Calculate preemptible savings
            preemptible_savings = preemptible_hours * (
                self.pricing["cpu_per_vcpu_hour"] + 
                self.pricing["memory_per_gb_hour"]
            ) * self.pricing["preemptible_discount"] * cost_factor
            
            total_hourly_cost = (cpu_cost + memory_cost + request_cost) / (days * 24)
            monthly_projection = total_hourly_cost * 24 * 30
            
            return CostMetrics(
                service_name=service_name,
                region=region,
                hourly_cost=total_hourly_cost,
                monthly_projection=monthly_projection,
                cpu_cost=cpu_cost,
                memory_cost=memory_cost,
                request_cost=request_cost,
                instance_hours=instance_hours,
                preemptible_savings=preemptible_savings,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing costs for {service_name} in {region}: {e}")
            return None
    
    async def _get_usage_metrics(
        self, 
        service_name: str, 
        region: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict:
        """Get usage metrics from Cloud Monitoring"""
        try:
            project_name = f"projects/{self.project_id}"
            interval = monitoring_v3.TimeInterval({
                "end_time": {"seconds": int(end_time.timestamp())},
                "start_time": {"seconds": int(start_time.timestamp())}
            })
            
            metrics = {}
            
            # CPU utilization metrics
            cpu_filter = (
                f'resource.type="cloud_run_revision" AND '
                f'resource.labels.service_name="{service_name}" AND '
                f'resource.labels.location="{region}" AND '
                f'metric.type="run.googleapis.com/container/cpu/utilizations"'
            )
            
            cpu_results = self.monitoring_client.list_time_series(
                request={
                    "name": project_name,
                    "filter": cpu_filter,
                    "interval": interval,
                    "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
                }
            )
            
            # Process CPU metrics
            cpu_utilization_sum = 0
            cpu_data_points = 0
            for result in cpu_results:
                for point in result.points:
                    cpu_utilization_sum += point.value.double_value
                    cpu_data_points += 1
            
            avg_cpu_utilization = cpu_utilization_sum / cpu_data_points if cpu_data_points > 0 else 0
            
            # Estimate resource usage (simplified calculation)
            hours_in_period = (end_time - start_time).total_seconds() / 3600
            estimated_vcpus = 2.0  # Assume 2 vCPUs average
            estimated_memory_gb = 4.0  # Assume 4GB average
            
            metrics["cpu_hours"] = hours_in_period * estimated_vcpus * avg_cpu_utilization
            metrics["memory_gb_hours"] = hours_in_period * estimated_memory_gb
            metrics["total_requests"] = 10000  
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime, timedelta
import uuid

class AdvancedGeologicalAPI:
    """Advanced API endpoints for geological data and analysis"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Prometheus Geological Analysis API",
            description="Advanced geological simulation and monitoring API",
            version="2.0.0"
        )
        self.security = HTTPBearer()
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_middleware(self):
        """Setup API middleware for logging, CORS, rate limiting"""
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.gzip import GZipMiddleware
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/api/v2/seismic/analyze", response_model=SeismicAnalysisResponse)
        async def analyze_seismic_data(
            request: SeismicAnalysisRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Advanced seismic data analysis endpoint"""
            
            # Validate authentication
            user_id = await self._validate_token(credentials.credentials)
            
            # Generate analysis ID
            analysis_id = str(uuid.uuid4())
            
            try:
                # Initialize seismic processor
                processor = AdvancedSeismicProcessor(sampling_rate=request.sampling_rate)
                
                # Process seismic data
                analysis_result = processor.process_seismic_stream(
                    data=np.array(request.waveform_data),
                    station_info=request.station_info.dict()
                )
                
                # Add metadata
                analysis_result.update({
                    'analysis_id': analysis_id,
                    'user_id': user_id,
                    'request_timestamp': datetime.now().isoformat(),
                    'processing_time_ms': 0  # Would be calculated
                })
                
                # Store results asynchronously
                background_tasks.add_task(
                    self._store_analysis_results,
                    analysis_id, analysis_result, 'seismic'
                )
                
                return SeismicAnalysisResponse(**analysis_result)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        @self.app.post("/api/v2/volcanic/monitor", response_model=VolcanicMonitoringResponse)
        async def monitor_volcanic_activity(
            request: VolcanicMonitoringRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Comprehensive volcanic monitoring endpoint"""
            
            user_id = await self._validate_token(credentials.credentials)
            monitoring_id = str(uuid.uuid4())
            
            try:
                # Initialize volcanic monitor
                monitor = AdvancedVolcanicMonitor()
                
                # Perform comprehensive assessment
                assessment = monitor.comprehensive_volcanic_assessment(
                    monitoring_data=request.monitoring_data,
                    volcano_info=request.volcano_info.dict()
                )
                
                # Add metadata
                assessment.update({
                    'monitoring_id': monitoring_id,
                    'user_id': user_id,
                    'request_timestamp': datetime.now().isoformat()
                })
                
                # Store results and trigger alerts if necessary
                background_tasks.add_task(
                    self._process_volcanic_assessment,
                    monitoring_id, assessment
                )
                
                return VolcanicMonitoringResponse(**assessment)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Monitoring failed: {str(e)}")
        
        @self.app.post("/api/v2/geochemical/analyze", response_model=GeochemicalAnalysisResponse)
        async def analyze_geochemical_composition(
            request: GeochemicalAnalysisRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Advanced geochemical composition analysis"""
            
            user_id = await self._validate_token(credentials.credentials)
            analysis_id = str(uuid.uuid4())
            
            try:
                # Initialize geochemical analyzer
                analyzer = AdvancedGeochemicalAnalyzer()
                
                # Perform analysis
                analysis_result = analyzer.analyze_rock_composition(
                    composition=request.composition,
                    temperature=request.temperature,
                    pressure=request.pressure
                )
                
                # Add metadata
                analysis_result.update({
                    'analysis_id': analysis_id,
                    'user_id': user_id,
                    'sample_id': request.sample_id,
                    'request_timestamp': datetime.now().isoformat()
                })
                
                # Store results
                background_tasks.add_task(
                    self._store_analysis_results,
                    analysis_id, analysis_result, 'geochemical'
                )
                
                return GeochemicalAnalysisResponse(**analysis_result)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Geochemical analysis failed: {str(e)}")
        
        @self.app.get("/api/v2/analysis/{analysis_id}", response_model=AnalysisStatusResponse)
        async def get_analysis_status(
            analysis_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get analysis status and results"""
            
            user_id = await self._validate_token(credentials.credentials)
            
            try:
                # Retrieve analysis from database/cache
                analysis_data = await self._get_analysis_results(analysis_id, user_id)
                
                if not analysis_data:
                    raise HTTPException(status_code=404, detail="Analysis not found")
                
                return AnalysisStatusResponse(**analysis_data)
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to retrieve analysis: {str(e)}")
        
        @self.app.get("/api/v2/spatial/query", response_model=SpatialQueryResponse)
        async def spatial_geological_query(
            bbox: str = Query(..., description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
            start_time: str = Query(..., description="Start time (ISO format)"),
            end_time: str = Query(..., description="End time (ISO format)"),
            data_types: List[str] = Query(["seismic"], description="Data types to include"),
            min_magnitude: float = Query(0.0, description="Minimum magnitude"),
            max_magnitude: float = Query(10.0, description="Maximum magnitude"),
            limit: int = Query(1000, le=10000, description="Maximum results"),
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Spatial query for geological data"""
            
            user_id = await self._validate_token(credentials.credentials)
            
            try:
                # Parse bounding box
                bbox_coords = [float(x) for x in bbox.split(',')]
                if len(bbox_coords) != 4:
                    raise ValueError("Invalid bounding box format")
                
                # Initialize database
                db = AdvancedGeologicalDatabase({
                    'postgres': {'host': 'localhost', 'database': 'geological_data'},
                    'influxdb': {'url': 'http://localhost:8086'}
                })
                
                # Perform spatial query
                results = db.query_spatial_events(
                    bbox=tuple(bbox_coords),
                    time_range=(start_time, end_time),
                    magnitude_range=(min_magnitude, max_magnitude)
                )
                
                # Limit results
                results = results[:limit]
                
                return SpatialQueryResponse(
                    query_id=str(uuid.uuid4()),
                    user_id=user_id,
                    bbox=bbox_coords,
                    time_range=[start_time, end_time],
                    result_count=len(results),
                    results=results,
                    query_timestamp=datetime.now().isoformat()
                )
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Spatial query failed: {str(e)}")
    
    async def _validate_token(self, token: str) -> str:
        """Validate JWT token and return user ID"""
        # Simplified token validation - implement proper JWT validation
        if not token or len(token) < 10:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Extract user ID from token (simplified)
        return f"user_{hash(token) % 10000}"
    
    async def _store_analysis_results(self, analysis_id: str, results: Dict[str, Any], analysis_type: str):
        """Store analysis results in database"""
        try:
            # Initialize database connection
            db = AdvancedGeologicalDatabase({
                'redis': {'host': 'localhost', 'port': 6379}
            })
            
            # Cache results
            db.cache_analysis_results(analysis_id, results, expiration_hours=24)
            
            logger.info(f"Stored {analysis_type} analysis results for {analysis_id}")
            
        except Exception as e:
            logger.error(f"Failed to store analysis results: {e}")
    
    async def _get_analysis_results(self, analysis_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis results from database"""
        try:
            db = AdvancedGeologicalDatabase({
                'redis': {'host': 'localhost', 'port': 6379}
            })
            
            results = db.get_cached_results(analysis_id)
            
            # Verify user has access to this analysis
            if results and results.get('user_id') == user_id:
                return results
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve analysis results: {e}")
            return None

# Pydantic models for request/response validation
class StationInfo(BaseModel):
    station_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    elevation_m: float
    network: str

class SeismicAnalysisRequest(BaseModel):
    waveform_data: List[List[float]]
    sampling_rate: float = Field(..., gt=0, le=1000)
    station_info: StationInfo
    analysis_parameters: Optional[Dict[str, Any]] = {}

class SeismicAnalysisResponse(BaseModel):
    analysis_id: str
    station_id: str
    processing_timestamp: str
    components: List[Dict[str, Any]]
    cross_component_analysis: Dict[str, Any]
    overall_quality: float

class VolcanicMonitoringRequest(BaseModel):
    volcano_info: Dict[str, Any]
    monitoring_data: Dict[str, Any]
    analysis_parameters: Optional[Dict[str, Any]] = {}

class VolcanicMonitoringResponse(BaseModel):
    monitoring_id: str
    volcano_id: str
    assessment_timestamp: str
    current_alert_level: str
    eruption_probability: Dict[str, float]
    recommendations: List[str]

            metrics["instance_hours"] = hours_in_period
            metrics["preemptible_hours"] = hours_in_period * 0.5  # Assume 50% preemptible
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting usage metrics: {e}")
            return {}
    
    def generate_optimization_recommendations(
        self, 
        cost_metrics: CostMetrics,
        strategy: CostOptimizationStrategy = CostOptimizationStrategy.BALANCED
    ) -> OptimizationRecommendation:
        """Generate cost optimization recommendations"""
        
        service_profile = self.service_profiles.get(cost_metrics.service_name, {})
        recommendations = []
        potential_savings = 0.0
        risk_level = "low"
        
        # Analyze current costs and usage patterns
        current_cost = cost_metrics.hourly_cost
        
        # Recommendation 1: Preemptible instances
        if strategy in [CostOptimizationStrategy.AGGRESSIVE, CostOptimizationStrategy.BALANCED]:
            preemptible_tolerance = service_profile.get("preemptible_tolerance", 0.3)
            if cost_metrics.preemptible_savings < current_cost * preemptible_tolerance:
                potential_preemptible_savings = current_cost * preemptible_tolerance * 0.3
                potential_savings += potential_preemptible_savings
                recommendations.append(
                    f"Increase preemptible instance usage to {preemptible_tolerance*100:.0f}% "
                    f"(save ~${potential_preemptible_savings*24*30:.2f}/month)"
                )
                if preemptible_tolerance > 0.5:
                    risk_level = "medium"
        
        # Recommendation 2: Resource right-sizing
        if cost_metrics.cpu_cost > cost_metrics.memory_cost * 2:
            # CPU-heavy workload
            cpu_savings = cost_metrics.cpu_cost * 0.2
            potential_savings += cpu_savings
            recommendations.append(
                f"Consider CPU optimization or workload distribution "
                f"(save ~${cpu_savings*24*30:.2f}/month)"
            )
        elif cost_metrics.memory_cost > cost_metrics.cpu_cost * 2:
            # Memory-heavy workload
            memory_savings = cost_metrics.memory_cost * 0.15
            potential_savings += memory_savings
            recommendations.append(
                f"Optimize memory usage or consider memory-optimized instances "
                f"(save ~${memory_savings*24*30:.2f}/month)"
            )
        
        # Recommendation 3: Scale-to-zero optimization
        if not service_profile.get("latency_sensitive", False):
            scale_to_zero_savings = current_cost * 0.3  # Assume 30% idle time
            potential_savings += scale_to_zero_savings
            recommendations.append(
                f"Enable scale-to-zero for idle periods "
                f"(save ~${scale_to_zero_savings*24*30:.2f}/month)"
            )
        
        # Recommendation 4: Regional cost arbitrage
        current_region_factor = self.regional_cost_factors.get(cost_metrics.region, 1.0)
        cheapest_region = min(self.regional_cost_factors.items(), key=lambda x: x[1])
        if cheapest_region[1] < current_region_factor:
            regional_savings = current_cost * (current_region_factor - cheapest_region[1])
            if regional_savings > current_cost * 0.05:  # Only recommend if >5% savings
                potential_savings += regional_savings
                recommendations.append(
                    f"Consider migrating to {cheapest_region[0]} for cost savings "
                    f"(save ~${regional_savings*24*30:.2f}/month)"
                )
                risk_level = "high"  # Migration has risks
        
        # Recommendation 5: Concurrency optimization
        if cost_metrics.request_cost > current_cost * 0.1:  # Request costs are significant
            concurrency_savings = cost_metrics.request_cost * 0.2
            potential_savings += concurrency_savings
            recommendations.append(
                f"Increase container concurrency to reduce instance count "
                f"(save ~${concurrency_savings*24*30:.2f}/month)"
            )
        
        # Calculate optimization priority
        savings_percentage = (potential_savings / current_cost) * 100 if current_cost > 0 else 0
        
        if savings_percentage > 30:
            priority = 1  # High priority
        elif savings_percentage > 15:
            priority = 2  # Medium-high priority
        elif savings_percentage > 5:
            priority = 3  # Medium priority
        else:
            priority = 4  # Low priority
        
        optimized_cost = current_cost - potential_savings
        
        return OptimizationRecommendation(
            service_name=cost_metrics.service_name,
            region=cost_metrics.region,
            current_cost=current_cost,
            optimized_cost=optimized_cost,
            savings_percentage=savings_percentage,
            recommendations=recommendations,
            risk_level=risk_level,
            implementation_priority=priority
        )
    
    def calculate_optimal_resources(
        self, 
        cost_metrics: CostMetrics,
        recommendation: OptimizationRecommendation
    ) -> ResourceOptimization:
        """Calculate optimal resource configuration"""
        
        service_profile = self.service_profiles.get(cost_metrics.service_name, {})
        
        # Base resource configuration
        if service_profile.get("cpu_intensive", False):
            cpu_limit = "2000m" if recommendation.risk_level == "low" else "4000m"
        else:
            cpu_limit = "1000m"
        
        if service_profile.get("memory_intensive", False):
            memory_limit = "4Gi" if recommendation.risk_level == "low" else "8Gi"
        else:
            memory_limit = "2Gi"
        
        # Concurrency based on service type
        if cost_metrics.service_name == "api-gateway":
            concurrency = 1000
        elif cost_metrics.service_name == "helios-core":
            concurrency = 10
        else:
            concurrency = 100
        
        # Instance scaling
        if service_profile.get("latency_sensitive", False):
            min_instances = 1
            max_instances = 50
        else:
            min_instances = 0
            max_instances = 100
        
        # Preemptible usage
        use_preemptible = recommendation.savings_percentage > 10 and recommendation.risk_level != "high"
        
        # Scale to zero
        scale_to_zero = not service_profile.get("latency_sensitive", False) and min_instances == 0
        
        # Timeout based on workload type
        if service_profile.get("batch_workload", False):
            timeout_seconds = 900  # 15 minutes for batch jobs
        elif cost_metrics.service_name == "helios-core":
            timeout_seconds = 300  # 5 minutes for AI processing
        else:
            timeout_seconds = 60   # 1 minute for API services
        
        return ResourceOptimization(
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
            min_instances=min_instances,
            max_instances=max_instances,
            concurrency=concurrency,
            use_preemptible=use_preemptible,
            scale_to_zero=scale_to_zero,
            timeout_seconds=timeout_seconds
        )
    
    async def apply_cost_optimizations(
        self, 
        service_name: str, 
        region: str,
        optimization: ResourceOptimization
    ) -> bool:
        """Apply cost optimization configuration to a Cloud Run service"""
        try:
            service_path = f"projects/{self.project_id}/locations/{region}/services/{service_name}"
            
            # Get current service
            service = self.run_client.get_service(name=service_path)
            
            # Update service configuration
            annotations = service.spec.template.metadata.annotations or {}
            
            # Update scaling configuration
            annotations["autoscaling.knative.dev/minScale"] = str(optimization.min_instances)
            annotations["autoscaling.knative.dev/maxScale"] = str(optimization.max_instances)
            
            # Update resource configuration
            if optimization.use_preemptible:
                annotations["run.googleapis.com/pricing-tier"] = "standard"
                annotations["run.googleapis.com/cpu-throttling"] = "true"
            
            # Update execution environment
            annotations["run.googleapis.com/execution-environment"] = "gen2"
            
            # Update container configuration
            service.spec.template.spec.container_concurrency = optimization.concurrency
            service.spec.template.spec.timeout_seconds = optimization.timeout_seconds
            
            # Update resource limits
            if service.spec.template.spec.containers:
                container = service.spec.template.spec.containers[0]
                container.resources.limits = {
                    "cpu": optimization.cpu_limit,
                    "memory": optimization.memory_limit
                }
            
            service.spec.template.metadata.annotations = annotations
            
            # Apply the update
            operation = self.run_client.update_service(service=service)
            result = operation.result(timeout=300)
            
            self.logger.info(f"Applied cost optimizations to {service_name} in {region}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying cost optimizations: {e}")
            return False
    
    async def optimize_all_services(
        self, 
        strategy: CostOptimizationStrategy = CostOptimizationStrategy.BALANCED
    ) -> Dict[str, List[OptimizationRecommendation]]:
        """Optimize costs for all services across all regions"""
        
        optimization_results = {}
        regions = list(self.regional_cost_factors.keys())[:3]  # Top 3 regions
        
        for service_name in self.service_profiles.keys():
            optimization_results[service_name] = []
            
            for region in regions:
                # Analyze current costs
                cost_metrics = await self.analyze_service_costs(service_name, region)
                
                if cost_metrics:
                    # Generate recommendations
                    recommendation = self.generate_optimization_recommendations(
                        cost_metrics, strategy
                    )
                    
                    # Calculate optimal resources
                    optimal_resources = self.calculate_optimal_resources(
                        cost_metrics, recommendation
                    )
                    
                    # Apply optimizations if savings are significant
                    if recommendation.savings_percentage > 5:  # Only apply if >5% savings
                        success = await self.apply_cost_optimizations(
                            service_name, region, optimal_resources
                        )
                        
                        if success:
                            optimization_results[service_name].append(recommendation)
        
        return optimization_results
    
    def generate_cost_report(
        self, 
        optimization_results: Dict[str, List[OptimizationRecommendation]]
    ) -> str:
        """Generate a comprehensive cost optimization report"""
        
        report = []
        report.append("# Cloud Run Cost Optimization Report")
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append("")
        
        total_current_cost = 0
        total_optimized_cost = 0
        total_savings = 0
        
        for service_name, recommendations in optimization_results.items():
            if not recommendations:
                continue
                
            report.append(f"## {service_name}")
            report.append("")
            
            service_current_cost = 0
            service_optimized_cost = 0
            
            for rec in recommendations:
                service_current_cost += rec.current_cost * 24 * 30  # Monthly cost
                service_optimized_cost += rec.optimized_cost * 24 * 30
                
                report.append(f"### {rec.region}")
                report.append(f"- **Current Monthly Cost**: ${rec.current_cost * 24 * 30:.2f}")
                report.append(f"- **Optimized Monthly Cost**: ${rec.optimized_cost * 24 * 30:.2f}")
                report.append(f"- **Potential Savings**: {rec.savings_percentage:.1f}% (${(rec.current_cost - rec.optimized_cost) * 24 * 30:.2f}/month)")
                report.append(f"- **Risk Level**: {rec.risk_level}")
                report.append(f"- **Priority**: {rec.implementation_priority}/5")
                report.append("")
                report.append("**Recommendations:**")
                for recommendation in rec.recommendations:
                    report.append(f"- {recommendation}")
                report.append("")
            
            total_current_cost += service_current_cost
            total_optimized_cost += service_optimized_cost
            
            service_savings = service_current_cost - service_optimized_cost
            total_savings += service_savings
            
            report.append(f"**Service Total Savings**: ${service_savings:.2f}/month")
            report.append("")
        
        # Summary
        overall_savings_percentage = (total_savings / total_current_cost) * 100 if total_current_cost > 0 else 0
        
        report.append("## Summary")
        report.append(f"- **Total Current Monthly Cost**: ${total_current_cost:.2f}")
        report.append(f"- **Total Optimized Monthly Cost**: ${total_optimized_cost:.2f}")
        report.append(f"- **Total Monthly Savings**: ${total_savings:.2f}")
        report.append(f"- **Overall Savings Percentage**: {overall_savings_percentage:.1f}%")
        report.append(f"- **Annual Savings Projection**: ${total_savings * 12:.2f}")
        
        return "\n".join(report)


async def main():
    """Main function for cost optimization"""
    logging.basicConfig(level=logging.INFO)
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "prometheus-ai-platform")
    billing_account_id = os.getenv("BILLING_ACCOUNT_ID", "")
    
    optimizer = CloudRunCostOptimizer(project_id, billing_account_id)
    
    # Run cost optimization
    print("Starting cost optimization analysis...")
    results = await optimizer.optimize_all_services(CostOptimizationStrategy.BALANCED)
    
    # Generate and print report
    report = optimizer.generate_cost_report(results)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())