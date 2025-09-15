"""
Advanced ETL Framework for Cross-Database Geological Data Synchronization

This module provides a comprehensive ETL (Extract, Transform, Load) framework
for synchronizing geological data across multiple database systems with
advanced transformation pipelines and data quality validation.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import asyncpg
import redis.asyncio as redis
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from neo4j import AsyncGraphDatabase
import structlog
from pydantic import BaseModel, Field, validator

logger = structlog.get_logger(__name__)


class DatabaseType(Enum):
    """Supported database types for ETL operations"""
    POSTGRESQL = "postgresql"
    INFLUXDB = "influxdb"
    NEO4J = "neo4j"
    REDIS = "redis"
    TIMESCALEDB = "timescaledb"


class TransformationType(Enum):
    """Types of data transformations available"""
    COORDINATE_TRANSFORM = "coordinate_transform"
    TEMPORAL_AGGREGATION = "temporal_aggregation"
    SPATIAL_INTERPOLATION = "spatial_interpolation"
    UNIT_CONVERSION = "unit_conversion"
    DATA_NORMALIZATION = "data_normalization"
    QUALITY_FILTERING = "quality_filtering"
    CAUSAL_RELATIONSHIP_EXTRACTION = "causal_relationship_extraction"
    GEOLOGICAL_CLASSIFICATION = "geological_classification"


@dataclass
class ETLJobConfig:
    """Configuration for ETL job execution"""
    job_id: str
    job_name: str
    source_config: Dict[str, Any]
    target_config: Dict[str, Any]
    transformation_pipeline: List[Dict[str, Any]]
    schedule_cron: Optional[str] = None
    retry_policy: Dict[str, Any] = None
    data_quality_checks: List[Dict[str, Any]] = None
    notification_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.retry_policy is None:
            self.retry_policy = {
                'max_retries': 3,
                'retry_delay': 60,
                'exponential_backoff': True
            }


@dataclass
class ETLJobResult:
    """Result of ETL job execution"""
    job_id: str
    execution_id: str
    start_time: datetime
    end_time: datetime
    status: str
    records_processed: int
    records_successful: int
    records_failed: int
    error_details: Optional[List[str]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class DatabaseConnector(ABC):
    """Abstract base class for database connectors"""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection"""
        pass
    
    @abstractmethod
    async def extract_data(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract data from database"""
        pass
    
    @abstractmethod
    async def load_data(self, data: pd.DataFrame, table: str, **kwargs) -> int:
        """Load data into database"""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """Execute arbitrary query"""
        pass


class PostgreSQLConnector(DatabaseConnector):
    """PostgreSQL/PostGIS database connector with advanced geospatial support"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = None
        self.connection = None
    
    async def connect(self) -> None:
        """Establish PostgreSQL connection"""
        self.engine = create_async_engine(self.connection_string)
        self.connection = await asyncpg.connect(self.connection_string)
        logger.info("Connected to PostgreSQL database")
    
    async def disconnect(self) -> None:
        """Close PostgreSQL connection"""
        if self.connection:
            await self.connection.close()
        if self.engine:
            await self.engine.dispose()
        logger.info("Disconnected from PostgreSQL database")
    
    async def extract_data(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract data from PostgreSQL with geospatial support"""
        try:
            if params:
                result = await self.connection.fetch(query, *params.values())
            else:
                result = await self.connection.fetch(query)
            
            # Convert to DataFrame
            if result:
                columns = list(result[0].keys())
                data = [list(row.values()) for row in result]
                df = pd.DataFrame(data, columns=columns)
                
                # Handle PostGIS geometry columns
                for col in df.columns:
                    if df[col].dtype == 'object' and any(isinstance(x, str) and x.startswith('POINT') for x in df[col].dropna()):
                        # Convert WKT geometry to separate lat/lon columns
                        df[f'{col}_lat'] = df[col].apply(self._extract_lat_from_wkt)
                        df[f'{col}_lon'] = df[col].apply(self._extract_lon_from_wkt)
                
                logger.debug("Extracted data from PostgreSQL", rows=len(df))
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error("Failed to extract data from PostgreSQL", error=str(e))
            raise
    
    async def load_data(self, data: pd.DataFrame, table: str, **kwargs) -> int:
        """Load data into PostgreSQL with upsert support"""
        try:
            if data.empty:
                return 0
            
            # Convert DataFrame to records
            records = data.to_dict('records')
            
            # Generate upsert query
            columns = list(data.columns)
            placeholders = ', '.join([f'${i+1}' for i in range(len(columns))])
            
            conflict_columns = kwargs.get('conflict_columns', ['id'])
            update_columns = [col for col in columns if col not in conflict_columns]
            
            upsert_query = f"""
                INSERT INTO {table} ({', '.join(columns)})
                VALUES ({placeholders})
                ON CONFLICT ({', '.join(conflict_columns)})
                DO UPDATE SET {', '.join([f'{col} = EXCLUDED.{col}' for col in update_columns])}
            """
            
            # Execute batch insert
            values_list = [list(record.values()) for record in records]
            await self.connection.executemany(upsert_query, values_list)
            
            logger.info("Loaded data into PostgreSQL", 
                       table=table, 
                       rows=len(data))
            return len(data)
            
        except Exception as e:
            logger.error("Failed to load data into PostgreSQL", 
                        table=table, 
                        error=str(e))
            raise
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """Execute PostgreSQL query"""
        try:
            if params:
                result = await self.connection.fetch(query, *params.values())
            else:
                result = await self.connection.fetch(query)
            return result
        except Exception as e:
            logger.error("Failed to execute PostgreSQL query", error=str(e))
            raise
    
    def _extract_lat_from_wkt(self, wkt_string: str) -> Optional[float]:
        """Extract latitude from WKT POINT string"""
        if not isinstance(wkt_string, str) or not wkt_string.startswith('POINT'):
            return None
        try:
            coords = wkt_string.replace('POINT(', '').replace(')', '').split()
            return float(coords[1])  # Latitude is second coordinate
        except:
            return None
    
    def _extract_lon_from_wkt(self, wkt_string: str) -> Optional[float]:
        """Extract longitude from WKT POINT string"""
        if not isinstance(wkt_string, str) or not wkt_string.startswith('POINT'):
            return None
        try:
            coords = wkt_string.replace('POINT(', '').replace(')', '').split()
            return float(coords[0])  # Longitude is first coordinate
        except:
            return None


class InfluxDBConnector(DatabaseConnector):
    """InfluxDB connector for time-series geological data"""
    
    def __init__(self, url: str, token: str, org: str, bucket: str):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.client = None
        self.query_api = None
        self.write_api = None
    
    async def connect(self) -> None:
        """Establish InfluxDB connection"""
        self.client = InfluxDBClientAsync(url=self.url, token=self.token, org=self.org)
        self.query_api = self.client.query_api()
        self.write_api = self.client.write_api()
        logger.info("Connected to InfluxDB")
    
    async def disconnect(self) -> None:
        """Close InfluxDB connection"""
        if self.client:
            await self.client.close()
        logger.info("Disconnected from InfluxDB")
    
    async def extract_data(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract time-series data from InfluxDB"""
        try:
            # Execute Flux query
            result = await self.query_api.query_data_frame(query)
            
            if isinstance(result, list) and len(result) > 0:
                df = pd.concat(result, ignore_index=True)
            elif isinstance(result, pd.DataFrame):
                df = result
            else:
                df = pd.DataFrame()
            
            # Convert _time column to datetime
            if '_time' in df.columns:
                df['_time'] = pd.to_datetime(df['_time'])
            
            logger.debug("Extracted data from InfluxDB", rows=len(df))
            return df
            
        except Exception as e:
            logger.error("Failed to extract data from InfluxDB", error=str(e))
            raise
    
    async def load_data(self, data: pd.DataFrame, table: str, **kwargs) -> int:
        """Load time-series data into InfluxDB"""
        try:
            if data.empty:
                return 0
            
            # Convert DataFrame to line protocol
            measurement = kwargs.get('measurement', table)
            time_column = kwargs.get('time_column', 'timestamp')
            tag_columns = kwargs.get('tag_columns', [])
            field_columns = kwargs.get('field_columns', [])
            
            # Auto-detect field columns if not specified
            if not field_columns:
                field_columns = [col for col in data.columns 
                               if col not in tag_columns and col != time_column
                               and pd.api.types.is_numeric_dtype(data[col])]
            
            # Prepare data for InfluxDB
            records = []
            for _, row in data.iterrows():
                record = {
                    'measurement': measurement,
                    'time': row[time_column] if time_column in row else datetime.utcnow(),
                    'tags': {col: str(row[col]) for col in tag_columns if col in row},
                    'fields': {col: float(row[col]) for col in field_columns if col in row and pd.notna(row[col])}
                }
                records.append(record)
            
            # Write to InfluxDB
            await self.write_api.write(bucket=self.bucket, record=records)
            
            logger.info("Loaded data into InfluxDB", 
                       measurement=measurement, 
                       rows=len(data))
            return len(data)
            
        except Exception as e:
            logger.error("Failed to load data into InfluxDB", 
                        measurement=table, 
                        error=str(e))
            raise
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """Execute InfluxDB Flux query"""
        try:
            result = await self.query_api.query(query)
            return result
        except Exception as e:
            logger.error("Failed to execute InfluxDB query", error=str(e))
            raise


class Neo4jConnector(DatabaseConnector):
    """Neo4j connector for geological relationship graphs"""
    
    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
    
    async def connect(self) -> None:
        """Establish Neo4j connection"""
        self.driver = AsyncGraphDatabase.driver(
            self.uri, 
            auth=(self.username, self.password)
        )
        logger.info("Connected to Neo4j database")
    
    async def disconnect(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
        logger.info("Disconnected from Neo4j database")
    
    async def extract_data(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract graph data from Neo4j"""
        try:
            async with self.driver.session() as session:
                result = await session.run(query, params or {})
                records = await result.data()
                
                if records:
                    df = pd.DataFrame(records)
                    logger.debug("Extracted data from Neo4j", rows=len(df))
                    return df
                else:
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error("Failed to extract data from Neo4j", error=str(e))
            raise
    
    async def load_data(self, data: pd.DataFrame, table: str, **kwargs) -> int:
        """Load graph data into Neo4j"""
        try:
            if data.empty:
                return 0
            
            node_label = kwargs.get('node_label', table)
            relationship_type = kwargs.get('relationship_type')
            
            async with self.driver.session() as session:
                if relationship_type:
                    # Load relationships
                    query = f"""
                        UNWIND $records AS record
                        MATCH (a:{node_label} {{id: record.source_id}})
                        MATCH (b:{node_label} {{id: record.target_id}})
                        MERGE (a)-[r:{relationship_type}]->(b)
                        SET r += record.properties
                    """
                else:
                    # Load nodes
                    query = f"""
                        UNWIND $records AS record
                        MERGE (n:{node_label} {{id: record.id}})
                        SET n += record.properties
                    """
                
                records = data.to_dict('records')
                await session.run(query, {'records': records})
                
                logger.info("Loaded data into Neo4j", 
                           label=node_label, 
                           rows=len(data))
                return len(data)
                
        except Exception as e:
            logger.error("Failed to load data into Neo4j", 
                        label=table, 
                        error=str(e))
            raise
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """Execute Neo4j Cypher query"""
        try:
            async with self.driver.session() as session:
                result = await session.run(query, params or {})
                return await result.data()
        except Exception as e:
            logger.error("Failed to execute Neo4j query", error=str(e))
            raise


class RedisConnector(DatabaseConnector):
    """Redis connector for caching and session management"""
    
    def __init__(self, url: str):
        self.url = url
        self.client = None
    
    async def connect(self) -> None:
        """Establish Redis connection"""
        self.client = redis.from_url(self.url)
        logger.info("Connected to Redis")
    
    async def disconnect(self) -> None:
        """Close Redis connection"""
        if self.client:
            await self.client.close()
        logger.info("Disconnected from Redis")
    
    async def extract_data(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract data from Redis (limited support)"""
        try:
            # Redis doesn't support complex queries, so we use pattern matching
            pattern = params.get('pattern', '*') if params else '*'
            keys = await self.client.keys(pattern)
            
            data = []
            for key in keys:
                value = await self.client.get(key)
                if value:
                    try:
                        # Try to parse as JSON
                        parsed_value = json.loads(value)
                        data.append({'key': key.decode(), 'value': parsed_value})
                    except:
                        # Store as string if not JSON
                        data.append({'key': key.decode(), 'value': value.decode()})
            
            df = pd.DataFrame(data)
            logger.debug("Extracted data from Redis", rows=len(df))
            return df
            
        except Exception as e:
            logger.error("Failed to extract data from Redis", error=str(e))
            raise
    
    async def load_data(self, data: pd.DataFrame, table: str, **kwargs) -> int:
        """Load data into Redis"""
        try:
            if data.empty:
                return 0
            
            key_column = kwargs.get('key_column', 'key')
            value_column = kwargs.get('value_column', 'value')
            ttl = kwargs.get('ttl')  # Time to live in seconds
            
            loaded_count = 0
            for _, row in data.iterrows():
                key = str(row[key_column])
                value = row[value_column]
                
                # Serialize value as JSON if it's not a string
                if not isinstance(value, str):
                    value = json.dumps(value, default=str)
                
                if ttl:
                    await self.client.setex(key, ttl, value)
                else:
                    await self.client.set(key, value)
                
                loaded_count += 1
            
            logger.info("Loaded data into Redis", 
                       prefix=table, 
                       rows=loaded_count)
            return loaded_count
            
        except Exception as e:
            logger.error("Failed to load data into Redis", error=str(e))
            raise
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """Execute Redis command"""
        try:
            # Redis doesn't have SQL-like queries, so we interpret the query as a command
            command_parts = query.split()
            if len(command_parts) >= 2:
                command = command_parts[0].upper()
                key = command_parts[1]
                
                if command == 'GET':
                    return await self.client.get(key)
                elif command == 'SET' and len(command_parts) >= 3:
                    value = ' '.join(command_parts[2:])
                    return await self.client.set(key, value)
                elif command == 'DEL':
                    return await self.client.delete(key)
            
            return None
        except Exception as e:
            logger.error("Failed to execute Redis command", error=str(e))
            raise


class DataTransformer:
    """Advanced data transformation engine for geological data"""
    
    def __init__(self):
        self.transformation_functions = {
            TransformationType.COORDINATE_TRANSFORM: self._coordinate_transform,
            TransformationType.TEMPORAL_AGGREGATION: self._temporal_aggregation,
            TransformationType.SPATIAL_INTERPOLATION: self._spatial_interpolation,
            TransformationType.UNIT_CONVERSION: self._unit_conversion,
            TransformationType.DATA_NORMALIZATION: self._data_normalization,
            TransformationType.QUALITY_FILTERING: self._quality_filtering,
            TransformationType.CAUSAL_RELATIONSHIP_EXTRACTION: self._causal_relationship_extraction,
            TransformationType.GEOLOGICAL_CLASSIFICATION: self._geological_classification
        }
    
    async def transform_data(self, 
                           data: pd.DataFrame, 
                           transformations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Apply series of transformations to data"""
        transformed_data = data.copy()
        
        for transformation in transformations:
            transform_type = TransformationType(transformation['type'])
            transform_params = transformation.get('params', {})
            
            if transform_type in self.transformation_functions:
                transformed_data = await self.transformation_functions[transform_type](
                    transformed_data, 
                    **transform_params
                )
                logger.debug("Applied transformation", 
                           type=transform_type.value,
                           rows_before=len(data),
                           rows_after=len(transformed_data))
            else:
                logger.warning("Unknown transformation type", type=transform_type.value)
        
        return transformed_data
    
    async def _coordinate_transform(self, 
                                  data: pd.DataFrame, 
                                  source_crs: str = 'EPSG:4326',
                                  target_crs: str = 'EPSG:3857',
                                  lat_col: str = 'latitude',
                                  lon_col: str = 'longitude') -> pd.DataFrame:
        """Transform coordinate reference system"""
        try:
            import pyproj
            
            transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
            
            if lat_col in data.columns and lon_col in data.columns:
                # Transform coordinates
                x_coords, y_coords = transformer.transform(
                    data[lon_col].values, 
                    data[lat_col].values
                )
                
                data[f'{lon_col}_transformed'] = x_coords
                data[f'{lat_col}_transformed'] = y_coords
            
            return data
            
        except Exception as e:
            logger.error("Coordinate transformation failed", error=str(e))
            return data
    
    async def _temporal_aggregation(self, 
                                  data: pd.DataFrame,
                                  time_col: str = 'timestamp',
                                  aggregation_period: str = '1H',
                                  aggregation_func: str = 'mean') -> pd.DataFrame:
        """Aggregate data over time periods"""
        try:
            if time_col not in data.columns:
                return data
            
            # Ensure time column is datetime
            data[time_col] = pd.to_datetime(data[time_col])
            
            # Set time column as index for resampling
            data_indexed = data.set_index(time_col)
            
            # Aggregate numeric columns
            numeric_columns = data_indexed.select_dtypes(include=[np.number]).columns
            
            if aggregation_func == 'mean':
                aggregated = data_indexed[numeric_columns].resample(aggregation_period).mean()
            elif aggregation_func == 'sum':
                aggregated = data_indexed[numeric_columns].resample(aggregation_period).sum()
            elif aggregation_func == 'max':
                aggregated = data_indexed[numeric_columns].resample(aggregation_period).max()
            elif aggregation_func == 'min':
                aggregated = data_indexed[numeric_columns].resample(aggregation_period).min()
            else:
                aggregated = data_indexed[numeric_columns].resample(aggregation_period).mean()
            
            # Reset index to get time column back
            aggregated = aggregated.reset_index()
            
            return aggregated
            
        except Exception as e:
            logger.error("Temporal aggregation failed", error=str(e))
            return data
    
    async def _spatial_interpolation(self, 
                                   data: pd.DataFrame,
                                   lat_col: str = 'latitude',
                                   lon_col: str = 'longitude',
                                   value_col: str = 'value',
                                   method: str = 'linear') -> pd.DataFrame:
        """Perform spatial interpolation of geological data"""
        try:
            from scipy.interpolate import griddata
            
            if not all(col in data.columns for col in [lat_col, lon_col, value_col]):
                return data
            
            # Remove rows with missing coordinates or values
            clean_data = data.dropna(subset=[lat_col, lon_col, value_col])
            
            if len(clean_data) < 3:  # Need at least 3 points for interpolation
                return data
            
            # Create coordinate arrays
            points = clean_data[[lon_col, lat_col]].values
            values = clean_data[value_col].values
            
            # Create regular grid for interpolation
            lon_min, lon_max = clean_data[lon_col].min(), clean_data[lon_col].max()
            lat_min, lat_max = clean_data[lat_col].min(), clean_data[lat_col].max()
            
            grid_lon, grid_lat = np.meshgrid(
                np.linspace(lon_min, lon_max, 50),
                np.linspace(lat_min, lat_max, 50)
            )
            
            # Interpolate values
            interpolated_values = griddata(
                points, values, (grid_lon, grid_lat), method=method
            )
            
            # Create DataFrame with interpolated data
            interpolated_data = pd.DataFrame({
                lon_col: grid_lon.flatten(),
                lat_col: grid_lat.flatten(),
                f'{value_col}_interpolated': interpolated_values.flatten()
            })
            
            # Remove NaN values
            interpolated_data = interpolated_data.dropna()
            
            return interpolated_data
            
        except Exception as e:
            logger.error("Spatial interpolation failed", error=str(e))
            return data
    
    async def _unit_conversion(self, 
                             data: pd.DataFrame,
                             conversions: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Convert units for geological measurements"""
        try:
            converted_data = data.copy()
            
            for column, conversion_config in conversions.items():
                if column in converted_data.columns:
                    from_unit = conversion_config.get('from_unit')
                    to_unit = conversion_config.get('to_unit')
                    factor = conversion_config.get('factor', 1.0)
                    offset = conversion_config.get('offset', 0.0)
                    
                    # Apply conversion: new_value = (old_value * factor) + offset
                    converted_data[f'{column}_{to_unit}'] = (
                        converted_data[column] * factor + offset
                    )
                    
                    logger.debug("Converted units", 
                               column=column,
                               from_unit=from_unit,
                               to_unit=to_unit)
            
            return converted_data
            
        except Exception as e:
            logger.error("Unit conversion failed", error=str(e))
            return data
    
    async def _data_normalization(self, 
                                data: pd.DataFrame,
                                method: str = 'z_score',
                                columns: List[str] = None) -> pd.DataFrame:
        """Normalize geological data values"""
        try:
            normalized_data = data.copy()
            
            # Select columns to normalize
            if columns is None:
                columns = normalized_data.select_dtypes(include=[np.number]).columns.tolist()
            
            for column in columns:
                if column in normalized_data.columns:
                    if method == 'z_score':
                        # Z-score normalization
                        mean_val = normalized_data[column].mean()
                        std_val = normalized_data[column].std()
                        if std_val > 0:
                            normalized_data[f'{column}_normalized'] = (
                                (normalized_data[column] - mean_val) / std_val
                            )
                    elif method == 'min_max':
                        # Min-max normalization
                        min_val = normalized_data[column].min()
                        max_val = normalized_data[column].max()
                        if max_val > min_val:
                            normalized_data[f'{column}_normalized'] = (
                                (normalized_data[column] - min_val) / (max_val - min_val)
                            )
                    elif method == 'robust':
                        # Robust normalization using median and IQR
                        median_val = normalized_data[column].median()
                        q75 = normalized_data[column].quantile(0.75)
                        q25 = normalized_data[column].quantile(0.25)
                        iqr = q75 - q25
                        if iqr > 0:
                            normalized_data[f'{column}_normalized'] = (
                                (normalized_data[column] - median_val) / iqr
                            )
            
            return normalized_data
            
        except Exception as e:
            logger.error("Data normalization failed", error=str(e))
            return data
    
    async def _quality_filtering(self, 
                               data: pd.DataFrame,
                               quality_col: str = 'quality_score',
                               min_quality: float = 0.7,
                               remove_outliers: bool = True,
                               outlier_method: str = 'iqr') -> pd.DataFrame:
        """Filter data based on quality metrics"""
        try:
            filtered_data = data.copy()
            
            # Filter by quality score
            if quality_col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[quality_col] >= min_quality]
                logger.debug("Applied quality filter", 
                           min_quality=min_quality,
                           rows_remaining=len(filtered_data))
            
            # Remove outliers
            if remove_outliers:
                numeric_columns = filtered_data.select_dtypes(include=[np.number]).columns
                
                for column in numeric_columns:
                    if column == quality_col:
                        continue
                    
                    if outlier_method == 'iqr':
                        # IQR method
                        Q1 = filtered_data[column].quantile(0.25)
                        Q3 = filtered_data[column].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        filtered_data = filtered_data[
                            (filtered_data[column] >= lower_bound) & 
                            (filtered_data[column] <= upper_bound)
                        ]
                    elif outlier_method == 'z_score':
                        # Z-score method
                        z_scores = np.abs((filtered_data[column] - filtered_data[column].mean()) / 
                                        filtered_data[column].std())
                        filtered_data = filtered_data[z_scores < 3]
            
            return filtered_data
            
        except Exception as e:
            logger.error("Quality filtering failed", error=str(e))
            return data
    
    async def _causal_relationship_extraction(self, 
                                            data: pd.DataFrame,
                                            time_col: str = 'timestamp',
                                            variables: List[str] = None,
                                            lag_window: int = 10) -> pd.DataFrame:
        """Extract causal relationships from time-series data"""
        try:
            if time_col not in data.columns:
                return pd.DataFrame()
            
            # Sort by time
            data_sorted = data.sort_values(time_col)
            
            # Select variables for causal analysis
            if variables is None:
                variables = data_sorted.select_dtypes(include=[np.number]).columns.tolist()
                variables = [col for col in variables if col != time_col]
            
            causal_relationships = []
            
            # Simple Granger causality test for each pair of variables
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i != j and var1 in data_sorted.columns and var2 in data_sorted.columns:
                        # Calculate correlation with lag
                        correlations = []
                        for lag in range(1, lag_window + 1):
                            if len(data_sorted) > lag:
                                lagged_var1 = data_sorted[var1].shift(lag)
                                correlation = lagged_var1.corr(data_sorted[var2])
                                if not np.isnan(correlation):
                                    correlations.append({
                                        'source_variable': var1,
                                        'target_variable': var2,
                                        'lag': lag,
                                        'correlation': correlation,
                                        'abs_correlation': abs(correlation)
                                    })
                        
                        # Find best lag
                        if correlations:
                            best_correlation = max(correlations, key=lambda x: x['abs_correlation'])
                            if best_correlation['abs_correlation'] > 0.3:  # Threshold for significance
                                causal_relationships.append(best_correlation)
            
            return pd.DataFrame(causal_relationships)
            
        except Exception as e:
            logger.error("Causal relationship extraction failed", error=str(e))
            return pd.DataFrame()
    
    async def _geological_classification(self, 
                                       data: pd.DataFrame,
                                       classification_rules: Dict[str, Any]) -> pd.DataFrame:
        """Classify geological data based on domain rules"""
        try:
            classified_data = data.copy()
            
            # Apply classification rules
            for rule_name, rule_config in classification_rules.items():
                conditions = rule_config.get('conditions', [])
                target_column = rule_config.get('target_column', f'{rule_name}_class')
                default_value = rule_config.get('default_value', 'unknown')
                
                # Initialize target column with default value
                classified_data[target_column] = default_value
                
                # Apply conditions
                for condition in conditions:
                    column = condition.get('column')
                    operator = condition.get('operator', '>=')
                    value = condition.get('value')
                    class_label = condition.get('class_label')
                    
                    if column in classified_data.columns:
                        if operator == '>=':
                            mask = classified_data[column] >= value
                        elif operator == '<=':
                            mask = classified_data[column] <= value
                        elif operator == '>':
                            mask = classified_data[column] > value
                        elif operator == '<':
                            mask = classified_data[column] < value
                        elif operator == '==':
                            mask = classified_data[column] == value
                        elif operator == '!=':
                            mask = classified_data[column] != value
                        else:
                            continue
                        
                        classified_data.loc[mask, target_column] = class_label
            
            return classified_data
            
        except Exception as e:
            logger.error("Geological classification failed", error=str(e))
            return data


# Continue with ETL job execution engine in next part...
class ETLJobExecutor:
    """Advanced ETL job execution engine with monitoring and error handling"""
    
    def __init__(self):
        self.connectors: Dict[str, DatabaseConnector] = {}
        self.transformer = DataTransformer()
        self.job_history: List[ETLJobResult] = []
        self.active_jobs: Dict[str, asyncio.Task] = {}
    
    def register_connector(self, name: str, connector: DatabaseConnector) -> None:
        """Register database connector"""
        self.connectors[name] = connector
        logger.info("Registered database connector", name=name)
    
    async def execute_etl_job(self, job_config: ETLJobConfig) -> ETLJobResult:
        """Execute ETL job with comprehensive monitoring"""
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        logger.info("Starting ETL job", 
                   job_id=job_config.job_id,
                   execution_id=execution_id)
        
        try:
            # Extract data from source
            source_connector = self.connectors[job_config.source_config['connector']]
            await source_connector.connect()
            
            extracted_data = await source_connector.extract_data(
                job_config.source_config['query'],
                job_config.source_config.get('params')
            )
            
            logger.info("Data extraction completed", 
                       rows=len(extracted_data))
            
            # Apply transformations
            transformed_data = await self.transformer.transform_data(
                extracted_data,
                job_config.transformation_pipeline
            )
            
            logger.info("Data transformation completed", 
                       rows=len(transformed_data))
            
            # Apply data quality checks
            if job_config.data_quality_checks:
                quality_results = await self._apply_quality_checks(
                    transformed_data,
                    job_config.data_quality_checks
                )
                
                # Filter out failed quality checks if configured
                if quality_results['failed_rows']:
                    logger.warning("Data quality issues detected", 
                                 failed_rows=len(quality_results['failed_rows']))
            
            # Load data to target
            target_connector = self.connectors[job_config.target_config['connector']]
            await target_connector.connect()
            
            loaded_count = await target_connector.load_data(
                transformed_data,
                job_config.target_config['table'],
                **job_config.target_config.get('options', {})
            )
            
            logger.info("Data loading completed", 
                       rows=loaded_count)
            
            # Cleanup connections
            await source_connector.disconnect()
            await target_connector.disconnect()
            
            # Create job result
            end_time = datetime.utcnow()
            job_result = ETLJobResult(
                job_id=job_config.job_id,
                execution_id=execution_id,
                start_time=start_time,
                end_time=end_time,
                status='success',
                records_processed=len(extracted_data),
                records_successful=loaded_count,
                records_failed=len(extracted_data) - loaded_count,
                performance_metrics={
                    'execution_time_seconds': (end_time - start_time).total_seconds(),
                    'records_per_second': loaded_count / max((end_time - start_time).total_seconds(), 1),
                    'data_size_mb': extracted_data.memory_usage(deep=True).sum() / 1024 / 1024
                }
            )
            
            self.job_history.append(job_result)
            
            logger.info("ETL job completed successfully", 
                       job_id=job_config.job_id,
                       execution_time=(end_time - start_time).total_seconds())
            
            return job_result
            
        except Exception as e:
            end_time = datetime.utcnow()
            error_message = str(e)
            
            logger.error("ETL job failed", 
                        job_id=job_config.job_id,
                        execution_id=execution_id,
                        error=error_message)
            
            # Create failed job result
            job_result = ETLJobResult(
                job_id=job_config.job_id,
                execution_id=execution_id,
                start_time=start_time,
                end_time=end_time,
                status='failed',
                records_processed=0,
                records_successful=0,
                records_failed=0,
                error_details=[error_message]
            )
            
            self.job_history.append(job_result)
            
            # Apply retry policy if configured
            if job_config.retry_policy and job_config.retry_policy.get('max_retries', 0) > 0:
                return await self._retry_job(job_config, job_result)
            
            return job_result
    
    async def _apply_quality_checks(self, 
                                  data: pd.DataFrame,
                                  quality_checks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply data quality validation checks"""
        failed_rows = []
        quality_metrics = {}
        
        for check in quality_checks:
            check_type = check.get('type')
            check_params = check.get('params', {})
            
            if check_type == 'completeness':
                # Check for missing values
                required_columns = check_params.get('required_columns', [])
                for column in required_columns:
                    if column in data.columns:
                        missing_count = data[column].isna().sum()
                        quality_metrics[f'{column}_completeness'] = 1 - (missing_count / len(data))
                        
                        if missing_count > 0:
                            failed_indices = data[data[column].isna()].index.tolist()
                            failed_rows.extend(failed_indices)
            
            elif check_type == 'range_validation':
                # Check value ranges
                column = check_params.get('column')
                min_value = check_params.get('min_value')
                max_value = check_params.get('max_value')
                
                if column in data.columns:
                    if min_value is not None:
                        invalid_mask = data[column] < min_value
                        failed_rows.extend(data[invalid_mask].index.tolist())
                    
                    if max_value is not None:
                        invalid_mask = data[column] > max_value
                        failed_rows.extend(data[invalid_mask].index.tolist())
            
            elif check_type == 'uniqueness':
                # Check for duplicate values
                columns = check_params.get('columns', [])
                if columns:
                    duplicates = data.duplicated(subset=columns, keep=False)
                    failed_rows.extend(data[duplicates].index.tolist())
                    quality_metrics['uniqueness_violations'] = duplicates.sum()
            
            elif check_type == 'format_validation':
                # Check data format patterns
                column = check_params.get('column')
                pattern = check_params.get('pattern')
                
                if column in data.columns and pattern:
                    import re
                    invalid_mask = ~data[column].astype(str).str.match(pattern)
                    failed_rows.extend(data[invalid_mask].index.tolist())
        
        return {
            'failed_rows': list(set(failed_rows)),
            'quality_metrics': quality_metrics
        }
    
    async def _retry_job(self, 
                        job_config: ETLJobConfig, 
                        failed_result: ETLJobResult) -> ETLJobResult:
        """Retry failed ETL job with exponential backoff"""
        retry_policy = job_config.retry_policy
        max_retries = retry_policy.get('max_retries', 3)
        base_delay = retry_policy.get('retry_delay', 60)
        exponential_backoff = retry_policy.get('exponential_backoff', True)
        
        for attempt in range(max_retries):
            # Calculate delay
            if exponential_backoff:
                delay = base_delay * (2 ** attempt)
            else:
                delay = base_delay
            
            logger.info("Retrying ETL job", 
                       job_id=job_config.job_id,
                       attempt=attempt + 1,
                       delay_seconds=delay)
            
            await asyncio.sleep(delay)
            
            # Retry job execution
            retry_result = await self.execute_etl_job(job_config)
            
            if retry_result.status == 'success':
                logger.info("ETL job retry successful", 
                           job_id=job_config.job_id,
                           attempt=attempt + 1)
                return retry_result
        
        logger.error("ETL job failed after all retries", 
                    job_id=job_config.job_id,
                    max_retries=max_retries)
        
        return failed_result
    
    async def schedule_etl_job(self, job_config: ETLJobConfig) -> None:
        """Schedule ETL job for periodic execution"""
        if not job_config.schedule_cron:
            logger.warning("No schedule specified for job", job_id=job_config.job_id)
            return
        
        # Simple cron-like scheduling (in production, use proper scheduler like APScheduler)
        logger.info("Scheduling ETL job", 
                   job_id=job_config.job_id,
                   schedule=job_config.schedule_cron)
        
        # Create background task for scheduled execution
        task = asyncio.create_task(self._run_scheduled_job(job_config))
        self.active_jobs[job_config.job_id] = task
    
    async def _run_scheduled_job(self, job_config: ETLJobConfig) -> None:
        """Run scheduled ETL job"""
        while True:
            try:
                # Execute job
                result = await self.execute_etl_job(job_config)
                
                # Send notifications if configured
                if job_config.notification_config:
                    await self._send_notification(job_config, result)
                
                # Calculate next execution time (simplified)
                # In production, use proper cron parsing
                await asyncio.sleep(3600)  # Run every hour for demo
                
            except asyncio.CancelledError:
                logger.info("Scheduled job cancelled", job_id=job_config.job_id)
                break
            except Exception as e:
                logger.error("Scheduled job error", 
                           job_id=job_config.job_id,
                           error=str(e))
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _send_notification(self, 
                               job_config: ETLJobConfig, 
                               result: ETLJobResult) -> None:
        """Send job completion notification"""
        notification_config = job_config.notification_config
        notification_type = notification_config.get('type', 'log')
        
        if notification_type == 'log':
            logger.info("ETL job notification", 
                       job_id=job_config.job_id,
                       status=result.status,
                       records_processed=result.records_processed)
        
        elif notification_type == 'webhook':
            # Send webhook notification
            import aiohttp
            
            webhook_url = notification_config.get('webhook_url')
            if webhook_url:
                payload = {
                    'job_id': job_config.job_id,
                    'execution_id': result.execution_id,
                    'status': result.status,
                    'records_processed': result.records_processed,
                    'execution_time': (result.end_time - result.start_time).total_seconds()
                }
                
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.post(webhook_url, json=payload) as response:
                            if response.status == 200:
                                logger.info("Webhook notification sent", 
                                           job_id=job_config.job_id)
                            else:
                                logger.error("Webhook notification failed", 
                                           job_id=job_config.job_id,
                                           status=response.status)
                    except Exception as e:
                        logger.error("Webhook notification error", 
                                   job_id=job_config.job_id,
                                   error=str(e))
    
    def get_job_history(self, job_id: Optional[str] = None) -> List[ETLJobResult]:
        """Get ETL job execution history"""
        if job_id:
            return [result for result in self.job_history if result.job_id == job_id]
        return self.job_history
    
    def get_job_statistics(self) -> Dict[str, Any]:
        """Get ETL job execution statistics"""
        if not self.job_history:
            return {}
        
        total_jobs = len(self.job_history)
        successful_jobs = len([r for r in self.job_history if r.status == 'success'])
        failed_jobs = total_jobs - successful_jobs
        
        total_records = sum(r.records_processed for r in self.job_history)
        avg_execution_time = sum(
            (r.end_time - r.start_time).total_seconds() 
            for r in self.job_history
        ) / total_jobs
        
        return {
            'total_jobs': total_jobs,
            'successful_jobs': successful_jobs,
            'failed_jobs': failed_jobs,
            'success_rate': successful_jobs / total_jobs if total_jobs > 0 else 0,
            'total_records_processed': total_records,
            'average_execution_time_seconds': avg_execution_time
        }


class DatabaseConnectorFactory:
    """Factory for creating database connectors"""
    
    @staticmethod
    def create_connector(db_type: DatabaseType, config: Dict[str, Any]) -> DatabaseConnector:
        """Create database connector based on type and configuration"""
        
        if db_type == DatabaseType.POSTGRESQL:
            connection_string = config.get('connection_string')
            if not connection_string:
                host = config.get('host', 'localhost')
                port = config.get('port', 5432)
                database = config.get('database', 'prometheus')
                username = config.get('username', 'prometheus')
                password = config.get('password', 'prometheus123')
                connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            
            return PostgreSQLConnector(connection_string)
        
        elif db_type == DatabaseType.INFLUXDB:
            url = config.get('url', 'http://localhost:8086')
            token = config.get('token', 'prometheus-token')
            org = config.get('org', 'prometheus')
            bucket = config.get('bucket', 'geological_data')
            
            return InfluxDBConnector(url, token, org, bucket)
        
        elif db_type == DatabaseType.NEO4J:
            uri = config.get('uri', 'bolt://localhost:7687')
            username = config.get('username', 'neo4j')
            password = config.get('password', 'prometheus123')
            
            return Neo4jConnector(uri, username, password)
        
        elif db_type == DatabaseType.REDIS:
            url = config.get('url', 'redis://localhost:6379')
            return RedisConnector(url)
        
        elif db_type == DatabaseType.TIMESCALEDB:
            # TimescaleDB uses PostgreSQL connector with specific configuration
            connection_string = config.get('connection_string')
            if not connection_string:
                host = config.get('host', 'localhost')
                port = config.get('port', 5432)
                database = config.get('database', 'timescaledb')
                username = config.get('username', 'prometheus')
                password = config.get('password', 'prometheus123')
                connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            
            return PostgreSQLConnector(connection_string)
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")


# Example usage and configuration
async def example_etl_pipeline():
    """Example of comprehensive ETL pipeline setup"""
    
    # Create ETL executor
    etl_executor = ETLJobExecutor()
    
    # Register database connectors
    postgres_connector = DatabaseConnectorFactory.create_connector(
        DatabaseType.POSTGRESQL,
        {
            'host': 'localhost',
            'port': 5432,
            'database': 'prometheus',
            'username': 'prometheus',
            'password': 'prometheus123'
        }
    )
    
    influx_connector = DatabaseConnectorFactory.create_connector(
        DatabaseType.INFLUXDB,
        {
            'url': 'http://localhost:8086',
            'token': 'prometheus-token',
            'org': 'prometheus',
            'bucket': 'geological_data'
        }
    )
    
    neo4j_connector = DatabaseConnectorFactory.create_connector(
        DatabaseType.NEO4J,
        {
            'uri': 'bolt://localhost:7687',
            'username': 'neo4j',
            'password': 'prometheus123'
        }
    )
    
    etl_executor.register_connector('postgres', postgres_connector)
    etl_executor.register_connector('influxdb', influx_connector)
    etl_executor.register_connector('neo4j', neo4j_connector)
    
    # Define ETL job configuration
    seismic_data_sync_job = ETLJobConfig(
        job_id='seismic_data_sync',
        job_name='Seismic Data Synchronization',
        source_config={
            'connector': 'postgres',
            'query': '''
                SELECT 
                    station_id,
                    timestamp,
                    latitude,
                    longitude,
                    magnitude,
                    depth_km,
                    quality_score
                FROM seismic_events 
                WHERE timestamp >= NOW() - INTERVAL '1 hour'
                AND quality_score >= 0.7
            '''
        },
        target_config={
            'connector': 'influxdb',
            'measurement': 'seismic_events',
            'options': {
                'time_column': 'timestamp',
                'tag_columns': ['station_id'],
                'field_columns': ['magnitude', 'depth_km', 'quality_score']
            }
        },
        transformation_pipeline=[
            {
                'type': 'coordinate_transform',
                'params': {
                    'source_crs': 'EPSG:4326',
                    'target_crs': 'EPSG:3857',
                    'lat_col': 'latitude',
                    'lon_col': 'longitude'
                }
            },
            {
                'type': 'quality_filtering',
                'params': {
                    'quality_col': 'quality_score',
                    'min_quality': 0.8,
                    'remove_outliers': True
                }
            },
            {
                'type': 'data_normalization',
                'params': {
                    'method': 'z_score',
                    'columns': ['magnitude', 'depth_km']
                }
            }
        ],
        data_quality_checks=[
            {
                'type': 'completeness',
                'params': {
                    'required_columns': ['station_id', 'timestamp', 'magnitude']
                }
            },
            {
                'type': 'range_validation',
                'params': {
                    'column': 'magnitude',
                    'min_value': -2.0,
                    'max_value': 10.0
                }
            }
        ],
        schedule_cron='0 * * * *',  # Every hour
        notification_config={
            'type': 'log'
        }
    )
    
    # Execute ETL job
    result = await etl_executor.execute_etl_job(seismic_data_sync_job)
    
    logger.info("ETL job completed", 
               job_id=result.job_id,
               status=result.status,
               records_processed=result.records_processed)
    
    # Get job statistics
    stats = etl_executor.get_job_statistics()
    logger.info("ETL statistics", **stats)


if __name__ == "__main__":
    asyncio.run(example_etl_pipeline())