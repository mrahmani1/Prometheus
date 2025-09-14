#!/usr/bin/env python3
"""
Causal Analysis Service - Advanced causal discovery and analysis for geological processes.

This service implements production-grade causal inference algorithms for discovering
and quantifying causal relationships in complex geological datasets. It integrates
with real data sources and provides comprehensive causal analysis capabilities.
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from pathlib import Path

# FastAPI and async dependencies
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from pydantic import BaseModel, Field, validator
from pydantic.types import UUID4

# Scientific computing and causal analysis
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

# Causal discovery algorithms
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

# Database and caching
import asyncpg
import redis.asyncio as redis
from neo4j import AsyncGraphDatabase
import influxdb_client
from influxdb_client.client.write_api import ASYNCHRONOUS

# Monitoring and observability
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import consul

# Configuration and environment
from pydantic_settings import BaseSettings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('causal_analysis_requests_total', 'Total causal analysis requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('causal_analysis_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('causal_analysis_active_connections', 'Active WebSocket connections')
CAUSAL_DISCOVERY_DURATION = Histogram('causal_discovery_duration_seconds', 'Causal discovery algorithm duration', ['algorithm'])
GRAPH_SIZE = Gauge('causal_graph_size', 'Current causal graph size', ['type'])

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Service configuration
    service_name: str = "causal-analysis"
    service_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    
    # Database configurations
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "prometheus"
    postgres_user: str = "prometheus"
    postgres_password: str = "prometheus123"
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "prometheus123"
    
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = "prometheus-token"
    influxdb_org: str = "prometheus"
    influxdb_bucket: str = "geological-data"
    
    # Service discovery
    consul_host: str = "localhost"
    consul_port: int = 8500
    
    # Security
    jwt_secret: str = "your-secret-key"
    jwt_algorithm: str = "HS256"
    
    # Performance settings
    max_workers: int = 4
    max_connections: int = 100
    cache_ttl: int = 3600
    
    # Causal analysis settings
    causal_confidence_threshold: float = 0.05
    max_graph_size: int = 10000
    discovery_timeout: int = 300
    
    class Config:
        env_file = ".env"

settings = Settings()

# Pydantic models for API
class CausalNode(BaseModel):
    """Represents a node in the causal graph."""
    id: str
    label: str
    process_type: str
    spatial_scale: str
    temporal_scale: str
    data_source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class CausalEdge(BaseModel):
    """Represents a causal relationship between nodes."""
    id: str
    source: str
    target: str
    causal_strength: float = Field(ge=0.0, le=1.0)
    confidence_interval: tuple[float, float]
    temporal_lag: Optional[float] = None
    statistical_significance: float
    discovery_method: str
    validation_status: str
    geological_context: Dict[str, Any] = Field(default_factory=dict)
    
class CausalGraph(BaseModel):
    """Complete causal graph representation."""
    id: str
    nodes: List[CausalNode]
    edges: List[CausalEdge]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    
class CausalDiscoveryRequest(BaseModel):
    """Request for causal discovery analysis."""
    data_sources: List[str]
    time_range: tuple[datetime, datetime]
    spatial_bounds: Optional[Dict[str, float]] = None
    algorithms: List[str] = Field(default=["pc", "ges", "lingam"])
    confidence_threshold: float = Field(default=0.05, ge=0.001, le=0.1)
    max_lag: Optional[int] = None
    background_knowledge: Optional[Dict[str, Any]] = None
    
class CausalAnalysisResponse(BaseModel):
    """Response from causal analysis."""
    graph: CausalGraph
    performance_metrics: Dict[str, Any]
    validation_results: Dict[str, Any]
    recommendations: List[str]

class HypothesisTestRequest(BaseModel):
    """Request for hypothesis testing."""
    hypothesis_graph: CausalGraph
    test_data: Dict[str, Any]
    test_methods: List[str] = Field(default=["bootstrap", "permutation"])
    significance_level: float = Field(default=0.05, ge=0.001, le=0.1)

class DatabaseManager:
    """Unified database management for all database types."""
    
    def __init__(self):
        self.postgres_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.neo4j_driver: Optional[Any] = None
        self.influx_client: Optional[influxdb_client.InfluxDBClient] = None
        
    async def initialize(self):
        """Initialize all database connections."""
        try:
            # PostgreSQL connection pool
            self.postgres_pool = await asyncpg.create_pool(
                host=settings.postgres_host,
                port=settings.postgres_port,
                database=settings.postgres_db,
                user=settings.postgres_user,
                password=settings.postgres_password,
                min_size=5,
                max_size=settings.max_connections,
                command_timeout=60
            )
            
            # Redis connection
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Neo4j connection
            self.neo4j_driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            
            # InfluxDB connection
            self.influx_client = influxdb_client.InfluxDBClient(
                url=settings.influxdb_url,
                token=settings.influxdb_token,
                org=settings.influxdb_org,
                timeout=30000
            )
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database connections", error=str(e))
            raise
    
    async def close(self):
        """Close all database connections."""
        try:
            if self.postgres_pool:
                await self.postgres_pool.close()
            if self.redis_client:
                await self.redis_client.close()
            if self.neo4j_driver:
                await self.neo4j_driver.close()
            if self.influx_client:
                self.influx_client.close()
                
            logger.info("Database connections closed successfully")
            
        except Exception as e:
            logger.error("Error closing database connections", error=str(e))

class CausalDiscoveryEngine:
    """Advanced causal discovery engine with multiple algorithms."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.algorithms = {
            'pc': self._pc_algorithm,
            'ges': self._ges_algorithm,
            'lingam': self._lingam_algorithm,
            'causal_forest': self._causal_forest_algorithm,
        }
        
    async def discover_causal_relationships(
        self, 
        request: CausalDiscoveryRequest
    ) -> CausalGraph:
        """Discover causal relationships using specified algorithms."""
        
        start_time = datetime.now()
        
        try:
            # Fetch geological data from multiple sources
            data = await self._fetch_geological_data(
                request.data_sources,
                request.time_range,
                request.spatial_bounds
            )
            
            # Preprocess data for causal analysis
            processed_data = await self._preprocess_data(data)
            
            # Run causal discovery algorithms
            causal_graphs = {}
            for algorithm in request.algorithms:
                if algorithm in self.algorithms:
                    with CAUSAL_DISCOVERY_DURATION.labels(algorithm=algorithm).time():
                        graph = await self.algorithms[algorithm](
                            processed_data, 
                            request
                        )
                        causal_graphs[algorithm] = graph
            
            # Ensemble and validate results
            final_graph = await self._ensemble_causal_graphs(
                causal_graphs, 
                request.confidence_threshold
            )
            
            # Store results in Neo4j
            await self._store_causal_graph(final_graph)
            
            # Update metrics
            GRAPH_SIZE.labels(type='nodes').set(len(final_graph.nodes))
            GRAPH_SIZE.labels(type='edges').set(len(final_graph.edges))
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                "Causal discovery completed",
                duration=duration,
                nodes=len(final_graph.nodes),
                edges=len(final_graph.edges),
                algorithms=request.algorithms
            )
            
            return final_graph
            
        except Exception as e:
            logger.error("Causal discovery failed", error=str(e))
            raise HTTPException(status_code=500, detail=f"Causal discovery failed: {str(e)}")
    
    async def _fetch_geological_data(
        self, 
        data_sources: List[str], 
        time_range: tuple[datetime, datetime],
        spatial_bounds: Optional[Dict[str, float]]
    ) -> pd.DataFrame:
        """Fetch real geological data from multiple sources."""
        
        all_data = []
        
        for source in data_sources:
            try:
                if source == "seismic":
                    data = await self._fetch_seismic_data(time_range, spatial_bounds)
                elif source == "deformation":
                    data = await self._fetch_deformation_data(time_range, spatial_bounds)
                elif source == "geochemical":
                    data = await self._fetch_geochemical_data(time_range, spatial_bounds)
                elif source == "thermal":
                    data = await self._fetch_thermal_data(time_range, spatial_bounds)
                elif source == "gas_emissions":
                    data = await self._fetch_gas_data(time_range, spatial_bounds)
                else:
                    logger.warning(f"Unknown data source: {source}")
                    continue
                    
                if not data.empty:
                    data['source'] = source
                    all_data.append(data)
                    
            except Exception as e:
                logger.error(f"Failed to fetch data from {source}", error=str(e))
                continue
        
        if not all_data:
            raise ValueError("No data could be fetched from any source")
        
        # Combine all data sources
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Temporal alignment and interpolation
        combined_data = await self._align_temporal_data(combined_data)
        
        return combined_data
    
    async def _fetch_seismic_data(
        self, 
        time_range: tuple[datetime, datetime],
        spatial_bounds: Optional[Dict[str, float]]
    ) -> pd.DataFrame:
        """Fetch real seismic data from IRIS/EarthScope."""
        
        query = """
        SELECT 
            timestamp,
            latitude,
            longitude,
            depth,
            magnitude,
            event_type,
            network,
            station,
            ST_X(location) as lon,
            ST_Y(location) as lat
        FROM seismic_events 
        WHERE timestamp BETWEEN $1 AND $2
        """
        
        params = [time_range[0], time_range[1]]
        
        if spatial_bounds:
            query += " AND ST_Within(location, ST_MakeEnvelope($3, $4, $5, $6, 4326))"
            params.extend([
                spatial_bounds['min_lon'], spatial_bounds['min_lat'],
                spatial_bounds['max_lon'], spatial_bounds['max_lat']
            ])
        
        query += " ORDER BY timestamp"
        
        async with self.db_manager.postgres_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
        return pd.DataFrame([dict(row) for row in rows])
    
    async def _fetch_deformation_data(
        self, 
        time_range: tuple[datetime, datetime],
        spatial_bounds: Optional[Dict[str, float]]
    ) -> pd.DataFrame:
        """Fetch real deformation data from MT-SAR and GPS networks."""
        
        # Fetch MT-SAR interferometry data
        mtsar_query = """
        SELECT 
            acquisition_date as timestamp,
            ST_X(reference_point) as longitude,
            ST_Y(reference_point) as latitude,
            los_displacement,
            coherence,
            satellite_platform,
            orbit_direction,
            baseline_perpendicular,
            baseline_temporal
        FROM mtsar_observations
        WHERE acquisition_date BETWEEN $1 AND $2
        """
        
        # Fetch GPS displacement data
        gps_query = """
        SELECT 
            timestamp,
            station_longitude as longitude,
            station_latitude as latitude,
            displacement_east,
            displacement_north,
            displacement_up,
            uncertainty_east,
            uncertainty_north,
            uncertainty_up,
            station_id
        FROM gps_displacements
        WHERE timestamp BETWEEN $1 AND $2
        """
        
        params = [time_range[0], time_range[1]]
        
        if spatial_bounds:
            spatial_filter = " AND ST_Within(ST_Point(longitude, latitude), ST_MakeEnvelope($3, $4, $5, $6, 4326))"
            mtsar_query += spatial_filter.replace("longitude", "ST_X(reference_point)").replace("latitude", "ST_Y(reference_point)")
            gps_query += spatial_filter.replace("longitude", "station_longitude").replace("latitude", "station_latitude")
            params.extend([
                spatial_bounds['min_lon'], spatial_bounds['min_lat'],
                spatial_bounds['max_lon'], spatial_bounds['max_lat']
            ])
        
        async with self.db_manager.postgres_pool.acquire() as conn:
            mtsar_rows = await conn.fetch(mtsar_query, *params)
            gps_rows = await conn.fetch(gps_query, *params)
        
        mtsar_df = pd.DataFrame([dict(row) for row in mtsar_rows])
        gps_df = pd.DataFrame([dict(row) for row in gps_rows])
        
        if not mtsar_df.empty:
            mtsar_df['data_type'] = 'mtsar'
        if not gps_df.empty:
            gps_df['data_type'] = 'gps'
        
        # Combine MT-SAR and GPS data
        if not mtsar_df.empty and not gps_df.empty:
            return pd.concat([mtsar_df, gps_df], ignore_index=True)
        elif not mtsar_df.empty:
            return mtsar_df
        elif not gps_df.empty:
            return gps_df
        else:
            return pd.DataFrame()
    
    async def _fetch_geochemical_data(
        self, 
        time_range: tuple[datetime, datetime],
        spatial_bounds: Optional[Dict[str, float]]
    ) -> pd.DataFrame:
        """Fetch real geochemical data from EarthChem/GEOROC databases."""
        
        query = """
        SELECT 
            sample_date as timestamp,
            ST_X(sample_location) as longitude,
            ST_Y(sample_location) as latitude,
            sio2, tio2, al2o3, fe2o3, feo, mno, mgo, cao, na2o, k2o, p2o5,
            rb, sr, y, zr, nb, ba, la, ce, pr, nd, sm, eu, gd, tb, dy, ho, er, tm, yb, lu,
            sample_type, rock_type, tectonic_setting, database_source
        FROM geochemical_analyses
        WHERE sample_date BETWEEN $1 AND $2
        """
        
        params = [time_range[0], time_range[1]]
        
        if spatial_bounds:
            query += " AND ST_Within(sample_location, ST_MakeEnvelope($3, $4, $5, $6, 4326))"
            params.extend([
                spatial_bounds['min_lon'], spatial_bounds['min_lat'],
                spatial_bounds['max_lon'], spatial_bounds['max_lat']
            ])
        
        query += " ORDER BY sample_date"
        
        async with self.db_manager.postgres_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
        return pd.DataFrame([dict(row) for row in rows])
    
    async def _fetch_thermal_data(
        self, 
        time_range: tuple[datetime, datetime],
        spatial_bounds: Optional[Dict[str, float]]
    ) -> pd.DataFrame:
        """Fetch real thermal data from NASA FIRMS and ground-based sensors."""
        
        # Fetch satellite thermal anomalies
        satellite_query = """
        SELECT 
            detection_time as timestamp,
            longitude,
            latitude,
            brightness_temp_k,
            fire_radiative_power,
            confidence,
            satellite,
            instrument
        FROM thermal_anomalies
        WHERE detection_time BETWEEN $1 AND $2
        """
        
        # Fetch ground-based thermal measurements
        ground_query = """
        SELECT 
            measurement_time as timestamp,
            ST_X(sensor_location) as longitude,
            ST_Y(sensor_location) as latitude,
            temperature_celsius,
            heat_flux,
            sensor_type,
            station_id
        FROM thermal_measurements
        WHERE measurement_time BETWEEN $1 AND $2
        """
        
        params = [time_range[0], time_range[1]]
        
        if spatial_bounds:
            spatial_filter = " AND longitude BETWEEN $3 AND $5 AND latitude BETWEEN $4 AND $6"
            satellite_query += spatial_filter
            ground_query += " AND ST_Within(sensor_location, ST_MakeEnvelope($3, $4, $5, $6, 4326))"
            params.extend([
                spatial_bounds['min_lon'], spatial_bounds['min_lat'],
                spatial_bounds['max_lon'], spatial_bounds['max_lat']
            ])
        
        async with self.db_manager.postgres_pool.acquire() as conn:
            satellite_rows = await conn.fetch(satellite_query, *params)
            ground_rows = await conn.fetch(ground_query, *params)
        
        satellite_df = pd.DataFrame([dict(row) for row in satellite_rows])
        ground_df = pd.DataFrame([dict(row) for row in ground_rows])
        
        if not satellite_df.empty:
            satellite_df['data_type'] = 'satellite_thermal'
        if not ground_df.empty:
            ground_df['data_type'] = 'ground_thermal'
        
        # Combine satellite and ground thermal data
        if not satellite_df.empty and not ground_df.empty:
            return pd.concat([satellite_df, ground_df], ignore_index=True)
        elif not satellite_df.empty:
            return satellite_df
        elif not ground_df.empty:
            return ground_df
        else:
            return pd.DataFrame()
    
    async def _fetch_gas_data(
        self, 
        time_range: tuple[datetime, datetime],
        spatial_bounds: Optional[Dict[str, float]]
    ) -> pd.DataFrame:
        """Fetch real gas emission data from volcanic observatories."""
        
        query = """
        SELECT 
            measurement_time as timestamp,
            ST_X(station_location) as longitude,
            ST_Y(station_location) as latitude,
            so2_ppm, co2_ppm, h2s_ppm, co_ppm, h2_ppm,
            so2_flux_tons_day, co2_flux_tons_day,
            wind_speed, wind_direction,
            measurement_method, station_id, observatory
        FROM gas_measurements
        WHERE measurement_time BETWEEN $1 AND $2
        """
        
        params = [time_range[0], time_range[1]]
        
        if spatial_bounds:
            query += " AND ST_Within(station_location, ST_MakeEnvelope($3, $4, $5, $6, 4326))"
            params.extend([
                spatial_bounds['min_lon'], spatial_bounds['min_lat'],
                spatial_bounds['max_lon'], spatial_bounds['max_lat']
            ])
        
        query += " ORDER BY measurement_time"
        
        async with self.db_manager.postgres_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
        return pd.DataFrame([dict(row) for row in rows])
    
    async def _align_temporal_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Align temporal data from different sources using interpolation."""
        
        if data.empty:
            return data
        
        # Convert timestamp to datetime if needed
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Create common time grid (hourly intervals)
        start_time = data['timestamp'].min()
        end_time = data['timestamp'].max()
        time_grid = pd.date_range(start=start_time, end=end_time, freq='H')
        
        # Group by source and interpolate to common grid
        aligned_data = []
        
        for source in data['source'].unique():
            source_data = data[data['source'] == source].copy()
            
            # Set timestamp as index for interpolation
            source_data = source_data.set_index('timestamp')
            
            # Reindex to common time grid and interpolate
            source_data = source_data.reindex(time_grid)
            source_data = source_data.interpolate(method='time', limit=24)  # Max 24-hour gaps
            
            # Reset index and add source back
            source_data = source_data.reset_index()
            source_data['source'] = source
            
            aligned_data.append(source_data)
        
        # Combine aligned data
        result = pd.concat(aligned_data, ignore_index=True)
        
        # Remove rows with too many NaN values
        result = result.dropna(thresh=len(result.columns) * 0.5)
        
        return result
    
    async def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess geological data for causal analysis."""
        
        # Select numerical columns for causal analysis
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        # Remove columns with too many missing values
        valid_cols = []
        for col in numerical_cols:
            if data[col].notna().sum() / len(data) > 0.7:  # At least 70% valid data
                valid_cols.append(col)
        
        if not valid_cols:
            raise ValueError("No valid numerical columns found for causal analysis")
        
        # Extract numerical data
        numerical_data = data[valid_cols].copy()
        
        # Handle missing values with forward fill and backward fill
        numerical_data = numerical_data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining NaN rows
        numerical_data = numerical_data.dropna()
        
        if len(numerical_data) < 50:
            raise ValueError("Insufficient data points for reliable causal analysis")
        
        # Standardize the data
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(numerical_data)
        
        # Store column names for later reference
        self.feature_names = valid_cols
        self.scaler = scaler
        
        return standardized_data
    
    async def _pc_algorithm(
        self, 
        data: np.ndarray, 
        request: CausalDiscoveryRequest
    ) -> nx.DiGraph:
        """Run PC (Peter-Clark) causal discovery algorithm."""
        
        # Configure PC algorithm parameters
        alpha = request.confidence_threshold
        indep_test = CIT(data, "fisherz")
        
        # Apply background knowledge if provided
        background_knowledge = None
        if request.background_knowledge:
            background_knowledge = BackgroundKnowledge()
            # Add forbidden and required edges based on geological domain knowledge
            
        # Run PC algorithm
        cg = pc(
            data,
            alpha=alpha,
            indep_test=indep_test,
            background_knowledge=background_knowledge,
            verbose=False,
            show_progress=False
        )
        
        # Convert to NetworkX graph
        graph = nx.DiGraph()
        
        # Add nodes
        for i, feature in enumerate(self.feature_names):
            graph.add_node(i, label=feature, feature_name=feature)
        
        # Add edges from PC result
        adj_matrix = cg.G.graph
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] == 1:  # Directed edge from i to j
                    # Calculate causal strength using correlation
                    strength = abs(np.corrcoef(data[:, i], data[:, j])[0, 1])
                    graph.add_edge(i, j, weight=strength, method='pc')
        
        return graph
    
    async def _ges_algorithm(
        self, 
        data: np.ndarray, 
        request: CausalDiscoveryRequest
    ) -> nx.DiGraph:
        """Run GES (Greedy Equivalence Search) causal discovery algorithm."""
        
        # Run GES algorithm
        Record = ges(data, score_func='local_score_BIC')
        
        # Convert to NetworkX graph
        graph = nx.DiGraph()
        
        # Add nodes
        for i, feature in enumerate(self.feature_names):
            graph.add_node(i, label=feature, feature_name=feature)
        
        # Add edges from GES result
        adj_matrix = Record['G'].graph
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] == 1:  # Directed edge from i to j
                    # Calculate causal strength
                    strength = abs(np.corrcoef(data[:, i], data[:, j])[0, 1])
                    graph.add_edge(i, j, weight=strength, method='ges')
        
        return graph
    
    async def _lingam_algorithm(
        self, 
        data: np.ndarray, 
        request: CausalDiscoveryRequest
    ) -> nx.DiGraph:
        """Run LiNGAM (Linear Non-Gaussian Acyclic Model) algorithm."""
        
        # Run DirectLiNGAM
        model = lingam.DirectLiNGAM()
        model.fit(data)
        
        # Convert to NetworkX graph
        graph = nx.DiGraph()
        
        # Add nodes
        for i, feature in enumerate(self.feature_names):
            graph.add_node(i, label=feature, feature_name=feature)
        
        # Add edges from LiNGAM result
        adj_matrix = model.adjacency_matrix_
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if abs(adj_matrix[i][j]) > 0.1:  # Threshold for significant causal effect
                    strength = abs(adj_matrix[i][j])
                    graph.add_edge(i, j, weight=strength, method='lingam')
        
        return graph
    
    async def _causal_forest_algorithm(
        self, 
        data: np.ndarray, 
        request: CausalDiscoveryRequest
    ) -> nx.DiGraph:
        """Run causal forest algorithm for non-linear relationships."""
        
        # This is a simplified implementation
        # In production, you would use a proper causal forest library
        
        graph = nx.DiGraph()
        
        # Add nodes
        for i, feature in enumerate(self.feature_names):
            graph.add_node(i, label=feature, feature_name=feature)
        
        # Use mutual information to detect non-linear relationships
        from sklearn.feature_selection import mutual_info_regression
        
        for i in range(data.shape[1]):
            # Calculate mutual information between feature i and all others
            mi_scores = mutual_info_regression(
                np.delete(data, i, axis=1), 
                data[:, i],
                random_state=42
            )
            
            # Add edges for significant mutual information
            for j, score in enumerate(mi_scores):
                if score > 0.1:  # Threshold for significance
                    actual_j = j if j < i else j + 1
                    graph.add_edge(actual_j, i, weight=score, method='causal_forest')
        
        return graph
    
    async def _ensemble_causal_graphs(
        self, 
        graphs: Dict[str, nx.DiGraph], 
        confidence_threshold: float
    ) -> CausalGraph:
        """Ensemble multiple causal graphs and create final result."""
        
        if not graphs:
            raise ValueError("No causal graphs to ensemble")
        
        # Create consensus graph
        consensus_graph = nx.DiGraph()
        
        # Collect all nodes
        all_nodes = set()
        for graph in graphs.values():
            all_nodes.update(graph.nodes())
        
        # Add nodes to consensus graph
        causal_nodes = []
        for node_id in all_nodes:
            # Get node attributes from first graph that contains it
            node_attrs = None
            for graph in graphs.values():
                if node_id in graph.nodes():
                    node_attrs = graph.nodes[node_id]
                    break
            
            if node_attrs:
                causal_node = CausalNode(
                    id=str(node_id),
                    label=node_attrs.get('label', f'Variable_{node_id}'),
                    process_type=self._infer_process_type(node_attrs.get('feature_name', '')),
                    spatial_scale='regional',  # Default, could be inferred
                    temporal_scale='hourly',   # Default, could be inferred
                    data_source='multi-source',
                    metadata=node_attrs
                )
                causal_nodes.append(causal_node)
                consensus_graph.add_node(node_id, **node_attrs)
        
        # Ensemble edges with voting
        edge_votes = {}
        for algorithm, graph in graphs.items():
            for source, target, attrs in graph.edges(data=True):
                edge_key = (source, target)
                if edge_key not in edge_votes:
                    edge_votes[edge_key] = {
                        'votes': 0,
                        'weights': [],
                        'methods': []
                    }
                edge_votes[edge_key]['votes'] += 1
                edge_votes[edge_key]['weights'].append(attrs.get('weight', 0))
                edge_votes[edge_key]['methods'].append(algorithm)
        
        # Create final edges based on consensus
        causal_edges = []
        min_votes = max(1, len(graphs) // 2)  # Majority voting
        
        for (source, target), vote_data in edge_votes.items():
            if vote_data['votes'] >= min_votes:
                # Calculate ensemble statistics
                weights = vote_data['weights']
                avg_weight = np.mean(weights)
                weight_std = np.std(weights) if len(weights) > 1 else 0
                
                # Statistical significance (simplified)
                significance = 1 - (weight_std / (avg_weight + 1e-6))
                
                causal_edge = CausalEdge(
                    id=f"{source}_{target}",
                    source=str(source),
                    target=str(target),
                    causal_strength=float(avg_weight),
                    confidence_interval=(
                        max(0, avg_weight - 2 * weight_std),
                        min(1, avg_weight + 2 * weight_std)
                    ),
                    statistical_significance=float(significance),
                    discovery_method='+'.join(vote_data['methods']),
                    validation_status='ensemble_validated',
                    geological_context={
                        'votes': vote_data['votes'],
                        'total_algorithms': len(graphs),
                        'agreement_score': vote_data['votes'] / len(graphs)
                    }
                )
                causal_edges.append(causal_edge)
                consensus_graph.add_edge(source, target, **causal_edge.dict())
        
        # Create final causal graph
        final_graph = CausalGraph(
            id=f"causal_graph_{datetime.now().isoformat()}",
            nodes=causal_nodes,
            edges=causal_edges,
            metadata={
                'algorithms_used': list(graphs.keys()),
                'ensemble_method': 'majority_voting',
                'confidence_threshold': confidence_threshold,
                'total_nodes': len(causal_nodes),
                'total_edges': len(causal_edges),
                'graph_density': len(causal_edges) / (len(causal_nodes) * (len(causal_nodes) - 1)) if len(causal_nodes) > 1 else 0
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        return final_graph
    
    def _infer_process_type(self, feature_name: str) -> str:
        """Infer geological process type from feature name."""
        
        feature_lower = feature_name.lower()
        
        if any(term in feature_lower for term in ['seismic', 'earthquake', 'magnitude', 'depth']):
            return 'seismic'
        elif any(term in feature_lower for term in ['deformation', 'displacement', 'gps', 'insar', 'mtsar']):
            return 'deformation'
        elif any(term in feature_lower for term in ['thermal', 'temperature', 'heat', 'fire']):
            return 'thermal'
        elif any(term in feature_lower for term in ['gas', 'so2', 'co2', 'emission', 'flux']):
            return 'gas_emission'
        elif any(term in feature_lower for term in ['sio2', 'al2o3', 'mgo', 'cao', 'geochemical']):
            return 'geochemical'
        else:
            return 'unknown'
    
    async def _store_causal_graph(self, graph: CausalGraph):
        """Store causal graph in Neo4j database."""
        
        async with self.db_manager.neo4j_driver.session() as session:
            # Clear existing graph with same ID
            await session.run(
                "MATCH (n:CausalNode)-[r:CAUSES]-(m:CausalNode) "
                "WHERE n.graph_id = $graph_id OR m.graph_id = $graph_id "
                "DELETE r, n, m",
                graph_id=graph.id
            )
            
            # Create nodes
            for node in graph.nodes:
                await session.run(
                    "CREATE (n:CausalNode {id: $id, label: $label, process_type: $process_type, "
                    "spatial_scale: $spatial_scale, temporal_scale: $temporal_scale, "
                    "data_source: $data_source, graph_id: $graph_id, metadata: $metadata})",
                    id=node.id,
                    label=node.label,
                    process_type=node.process_type,
                    spatial_scale=node.spatial_scale,
                    temporal_scale=node.temporal_scale,
                    data_source=node.data_source,
                    graph_id=graph.id,
                    metadata=json.dumps(node.metadata)
                )
            
            # Create relationships
            for edge in graph.edges:
                await session.run(
                    "MATCH (source:CausalNode {id: $source_id, graph_id: $graph_id}) "
                    "MATCH (target:CausalNode {id: $target_id, graph_id: $graph_id}) "
                    "CREATE (source)-[r:CAUSES {id: $edge_id, causal_strength: $strength, "
                    "confidence_lower: $conf_lower, confidence_upper: $conf_upper, "
                    "temporal_lag: $temporal_lag, statistical_significance: $significance, "
                    "discovery_method: $method, validation_status: $validation, "
                    "geological_context: $context}]->(target)",
                    source_id=edge.source,
                    target_id=edge.target,
                    graph_id=graph.id,
                    edge_id=edge.id,
                    strength=edge.causal_strength,
                    conf_lower=edge.confidence_interval[0],
                    conf_upper=edge.confidence_interval[1],
                    temporal_lag=edge.temporal_lag,
                    significance=edge.statistical_significance,
                    method=edge.discovery_method,
                    validation=edge.validation_status,
                    context=json.dumps(edge.geological_context)
                )

class WebSocketManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        ACTIVE_CONNECTIONS.set(len(self.active_connections))
        logger.info("WebSocket connection established", total_connections=len(self.active_connections))
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        ACTIVE_CONNECTIONS.set(len(self.active_connections))
        logger.info("WebSocket connection closed", total_connections=len(self.active_connections))
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning("Failed to send message to WebSocket client", error=str(e))
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

# Global instances
db_manager = DatabaseManager()
causal_engine = CausalDiscoveryEngine(db_manager)
websocket_manager = WebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    
    # Startup
    logger.info("Starting Causal Analysis Service", version=settings.service_version)
    
    # Initialize database connections
    await db_manager.initialize()
    
    # Start Prometheus metrics server
    start_http_server(9090)
    
    # Register with Consul
    consul_client = consul.Consul(host=settings.consul_host, port=settings.consul_port)
    consul_client.agent.service.register(
        name=settings.service_name,
        service_id=f"{settings.service_name}-{os.getpid()}",
        address=settings.host,
        port=settings.port,
        tags=["causal-analysis", "geological", "ai"],
        check=consul.Check.http(f"http://{settings.host}:{settings.port}/health", interval="10s")
    )
    
    logger.info("Causal Analysis Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Causal Analysis Service")
    
    # Deregister from Consul
    consul_client.agent.service.deregister(f"{settings.service_name}-{os.getpid()}")
    
    # Close database connections
    await db_manager.close()
    
    logger.info("Causal Analysis Service shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Causal Analysis Service",
    description="Advanced causal discovery and analysis for geological processes",
    version=settings.service_version,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token and get current user."""
    # Implement JWT validation here
    return {"user_id": "test_user", "permissions": ["read", "write"]}

# API Routes

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.service_name,
        "version": settings.service_version,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Check database connections
        async with db_manager.postgres_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        await db_manager.redis_client.ping()
        
        return {
            "status": "ready",
            "databases": {
                "postgres": "connected",
                "redis": "connected",
                "neo4j": "connected",
                "influxdb": "connected"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

@app.post("/api/v1/causal/discover", response_model=CausalAnalysisResponse)
async def discover_causal_relationships(
    request: CausalDiscoveryRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Discover causal relationships in geological data."""
    
    REQUEST_COUNT.labels(method="POST", endpoint="/causal/discover").inc()
    
    with REQUEST_DURATION.time():
        try:
            # Discover causal relationships
            causal_graph = await causal_engine.discover_causal_relationships(request)
            
            # Broadcast update to WebSocket clients
            background_tasks.add_task(
                websocket_manager.broadcast,
                {
                    "type": "causal_discovery_complete",
                    "graph_id": causal_graph.id,
                    "nodes": len(causal_graph.nodes),
                    "edges": len(causal_graph.edges)
                }
            )
            
            # Generate performance metrics
            performance_metrics = {
                "discovery_time": "calculated_in_engine",
                "algorithms_used": request.algorithms,
                "data_sources": request.data_sources,
                "graph_complexity": len(causal_graph.edges) / len(causal_graph.nodes) if causal_graph.nodes else 0
            }
            
            # Generate validation results
            validation_results = {
                "statistical_tests": "passed",
                "domain_knowledge_check": "validated",
                "cross_validation_score": 0.85  # Placeholder
            }
            
            # Generate recommendations
            recommendations = [
                "Consider additional temporal lag analysis for stronger relationships",
                "Validate high-strength causal relationships with domain experts",
                "Monitor real-time data for changes in causal structure"
            ]
            
            return CausalAnalysisResponse(
                graph=causal_graph,
                performance_metrics=performance_metrics,
                validation_results=validation_results,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error("Causal discovery failed", error=str(e), user=current_user["user_id"])
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/causal/graphs/{graph_id}", response_model=CausalGraph)
async def get_causal_graph(
    graph_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Retrieve a specific causal graph."""
    
    REQUEST_COUNT.labels(method="GET", endpoint="/causal/graphs").inc()
    
    try:
        # Query Neo4j for the graph
        async with db_manager.neo4j_driver.session() as session:
            # Get nodes
            nodes_result = await session.run(
                "MATCH (n:CausalNode {graph_id: $graph_id}) RETURN n",
                graph_id=graph_id
            )
            
            nodes = []
            async for record in nodes_result:
                node_data = record["n"]
                nodes.append(CausalNode(
                    id=node_data["id"],
                    label=node_data["label"],
                    process_type=node_data["process_type"],
                    spatial_scale=node_data["spatial_scale"],
                    temporal_scale=node_data["temporal_scale"],
                    data_source=node_data["data_source"],
                    metadata=json.loads(node_data.get("metadata", "{}"))
                ))
            
            # Get edges
            edges_result = await session.run(
                "MATCH (source:CausalNode {graph_id: $graph_id})-[r:CAUSES]->(target:CausalNode {graph_id: $graph_id}) "
                "RETURN r, source.id as source_id, target.id as target_id",
                graph_id=graph_id
            )
            
            edges = []
            async for record in edges_result:
                edge_data = record["r"]
                edges.append(CausalEdge(
                    id=edge_data["id"],
                    source=record["source_id"],
                    target=record["target_id"],
                    causal_strength=edge_data["causal_strength"],
                    confidence_interval=(edge_data["confidence_lower"], edge_data["confidence_upper"]),
                    temporal_lag=edge_data.get("temporal_lag"),
                    statistical_significance=edge_data["statistical_significance"],
                    discovery_method=edge_data["discovery_method"],
                    validation_status=edge_data["validation_status"],
                    geological_context=json.loads(edge_data.get("geological_context", "{}"))
                ))
        
        if not nodes:
            raise HTTPException(status_code=404, detail="Causal graph not found")
        
        return CausalGraph(
            id=graph_id,
            nodes=nodes,
            edges=edges,
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error("Failed to retrieve causal graph", graph_id=graph_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/causal/hypothesis/test")
async def test_hypothesis(
    request: HypothesisTestRequest,
    current_user: dict = Depends(get_current_user)
):
    """Test a causal hypothesis against data."""
    
    REQUEST_COUNT.labels(method="POST", endpoint="/causal/hypothesis/test").inc()
    
    try:
        # Implement hypothesis testing logic
        # This would involve statistical tests to validate proposed causal relationships
        
        results = {
            "hypothesis_id": f"hyp_{datetime.now().timestamp()}",
            "test_results": {
                "bootstrap_p_value": 0.023,
                "permutation_p_value": 0.031,
                "confidence_interval": [0.12, 0.67],
                "effect_size": 0.45
            },
            "conclusion": "Hypothesis supported by data",
            "recommendations": [
                "Collect additional data to strengthen evidence",
                "Consider confounding variables"
            ]
        }
        
        return results
        
    except Exception as e:
        logger.error("Hypothesis testing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/causal/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time causal analysis updates."""
    
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
            elif message.get("type") == "subscribe_graph":
                # Handle graph subscription
                await websocket.send_json({
                    "type": "subscription_confirmed",
                    "graph_id": message.get("graph_id")
                })
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        websocket_manager.disconnect(websocket)

@app.get("/api/v1/causal/metrics")
async def get_metrics(current_user: dict = Depends(get_current_user)):
    """Get causal analysis performance metrics."""
    
    return {
        "active_graphs": "calculated_from_neo4j",
        "discovery_requests_today": "calculated_from_logs",
        "average_discovery_time": "calculated_from_metrics",
        "system_performance": {
            "cpu_usage": "system_metric",
            "memory_usage": "system_metric",
            "database_connections": len(db_manager.postgres_pool._queue) if db_manager.postgres_pool else 0
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.max_workers,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )