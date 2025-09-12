#!/usr/bin/env python3
"""
Database Performance Testing for Prometheus AI Platform
Tests performance with large geological datasets across multiple database systems.
"""

import asyncio
import time
import psutil
import asyncpg
import redis
import influxdb_client
from neo4j import GraphDatabase
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance test results"""
    operation: str
    database: str
    duration: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    records_processed: int
    error_count: int
    timestamp: datetime

class GeologicalDataGenerator:
    """Generates realistic geological test data"""
    
    @staticmethod
    def generate_seismic_events(count: int) -> List[Dict]:
        """Generate synthetic seismic event data"""
        events = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(count):
            event = {
                'event_id': f'seismic_{i:06d}',
                'timestamp': base_time + timedelta(seconds=i * 60),
                'latitude': 40.7128 + np.random.normal(0, 0.1),
                'longitude': -74.0060 + np.random.normal(0, 0.1),
                'depth': np.random.exponential(15) + 1,  # Realistic depth distribution
                'magnitude': np.random.gamma(2, 0.8) + 1,  # Realistic magnitude distribution
                'station_count': np.random.randint(3, 15),
                'quality': np.random.choice(['A', 'B', 'C', 'D'], p=[0.4, 0.3, 0.2, 0.1]),
                'waveform_data': np.random.randn(1000).tolist(),  # Simulated waveform
            }
            events.append(event)
        
        return events
    
    @staticmethod
    def generate_deformation_data(count: int) -> List[Dict]:
        """Generate GPS/InSAR deformation data"""
        data = []
        base_time = datetime.now() - timedelta(days=365)
        
        for i in range(count):
            record = {
                'measurement_id': f'deform_{i:06d}',
                'timestamp': base_time + timedelta(hours=i * 6),  # 6-hour intervals
                'station_id': f'GPS_{i % 50:03d}',
                'latitude': 40.7128 + np.random.normal(0, 0.05),
                'longitude': -74.0060 + np.random.normal(0, 0.05),
                'elevation': 1500 + np.random.normal(0, 100),
                'displacement_north': np.random.normal(0, 0.01),
                'displacement_east': np.random.normal(0, 0.01),
                'displacement_up': np.random.normal(0, 0.005),
                'uncertainty_north': np.random.uniform(0.001, 0.005),
                'uncertainty_east': np.random.uniform(0.001, 0.005),
                'uncertainty_up': np.random.uniform(0.002, 0.008),
                'processing_quality': np.random.uniform(0.7, 1.0),
            }
            data.append(record)
        
        return data
    
    @staticmethod
    def generate_geological_model_data(count: int) -> List[Dict]:
        """Generate 3D geological model data"""
        data = []
        
        for i in range(count):
            record = {
                'voxel_id': f'voxel_{i:08d}',
                'x': np.random.uniform(-10000, 10000),
                'y': np.random.uniform(-10000, 10000),
                'z': np.random.uniform(-50000, 0),
                'rock_type': np.random.choice(['granite', 'basalt', 'sedimentary', 'metamorphic']),
                'density': np.random.uniform(2.0, 3.5),
                'porosity': np.random.uniform(0.01, 0.3),
                'permeability': np.random.exponential(1e-12),
                'temperature': 20 + abs(record['z']) * 0.025 if 'z' in locals() else 20,
                'pressure': 101325 + abs(record['z']) * 9800 * 2500 if 'z' in locals() else 101325,
                'mineral_composition': {
                    'quartz': np.random.uniform(0, 0.8),
                    'feldspar': np.random.uniform(0, 0.6),
                    'mica': np.random.uniform(0, 0.3),
                    'other': np.random.uniform(0, 0.2),
                },
            }
            data.append(record)
        
        return data

class DatabasePerformanceTester:
    """Main performance testing class for geological databases"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: List[PerformanceMetrics] = []
        self.data_generator = GeologicalDataGenerator()
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive database performance tests"""
        logger.info("Starting comprehensive database performance tests...")
        
        # Test configurations for different data sizes
        test_sizes = [1000, 10000, 100000, 500000]
        
        for size in test_sizes:
            logger.info(f"Testing with {size:,} records...")
            
            # Test PostgreSQL/PostGIS
            await self.test_postgresql_performance(size)
            
            # Test InfluxDB
            await self.test_influxdb_performance(size)
            
            # Test Neo4j
            await self.test_neo4j_performance(size)
            
            # Test Redis
            await self.test_redis_performance(size)
            
            # Test concurrent operations
            await self.test_concurrent_operations(size)
        
        # Generate performance report
        report = self.generate_performance_report()
        
        # Create visualizations
        self.create_performance_visualizations()
        
        return report
    
    async def test_postgresql_performance(self, record_count: int):
        """Test PostgreSQL/PostGIS performance with geological data"""
        logger.info(f"Testing PostgreSQL with {record_count:,} records...")
        
        # Connection setup
        conn = await asyncpg.connect(
            host=self.config['postgres']['host'],
            port=self.config['postgres']['port'],
            user=self.config['postgres']['user'],
            password=self.config['postgres']['password'],
            database=self.config['postgres']['database']
        )
        
        try:
            # Create test tables
            await self.setup_postgresql_tables(conn)
            
            # Test seismic data insertion
            seismic_data = self.data_generator.generate_seismic_events(record_count)
            await self.test_postgresql_insert(conn, 'seismic_events', seismic_data, 'seismic')
            
            # Test spatial queries
            await self.test_postgresql_spatial_queries(conn, record_count)
            
            # Test complex geological queries
            await self.test_postgresql_geological_queries(conn, record_count)
            
            # Test bulk operations
            await self.test_postgresql_bulk_operations(conn, record_count)
            
        finally:
            await conn.close()
    
    async def setup_postgresql_tables(self, conn):
        """Setup PostgreSQL tables for testing"""
        # Create seismic events table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS seismic_events (
                event_id VARCHAR(50) PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                location GEOMETRY(POINT, 4326),
                depth FLOAT NOT NULL,
                magnitude FLOAT NOT NULL,
                station_count INTEGER,
                quality CHAR(1),
                waveform_data JSONB
            )
        """)
        
        # Create spatial index
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_seismic_location 
            ON seismic_events USING GIST (location)
        """)
        
        # Create time index
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_seismic_timestamp 
            ON seismic_events (timestamp)
        """)
        
        # Create deformation data table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS deformation_data (
                measurement_id VARCHAR(50) PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                station_id VARCHAR(20),
                location GEOMETRY(POINT, 4326),
                elevation FLOAT,
                displacement_vector FLOAT[3],
                uncertainty_vector FLOAT[3],
                processing_quality FLOAT
            )
        """)
    
    async def test_postgresql_insert(self, conn, table: str, data: List[Dict], operation_type: str):
        """Test PostgreSQL insertion performance"""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        start_cpu = psutil.cpu_percent()
        
        try:
            if table == 'seismic_events':
                # Batch insert seismic events
                values = [
                    (
                        record['event_id'],
                        record['timestamp'],
                        f"POINT({record['longitude']} {record['latitude']})",
                        record['depth'],
                        record['magnitude'],
                        record['station_count'],
                        record['quality'],
                        json.dumps(record['waveform_data'])
                    )
                    for record in data
                ]
                
                await conn.executemany("""
                    INSERT INTO seismic_events 
                    (event_id, timestamp, location, depth, magnitude, station_count, quality, waveform_data)
                    VALUES ($1, $2, ST_GeomFromText($3, 4326), $4, $5, $6, $7, $8)
                """, values)
            
            duration = time.time() - start_time
            memory_used = psutil.virtual_memory().used - start_memory
            cpu_used = psutil.cpu_percent() - start_cpu
            
            metrics = PerformanceMetrics(
                operation=f'{operation_type}_insert',
                database='postgresql',
                duration=duration,
                throughput=len(data) / duration,
                memory_usage=memory_used,
                cpu_usage=cpu_used,
                records_processed=len(data),
                error_count=0,
                timestamp=datetime.now()
            )
            
            self.results.append(metrics)
            logger.info(f"PostgreSQL insert: {len(data):,} records in {duration:.2f}s ({metrics.throughput:.0f} records/s)")
            
        except Exception as e:
            logger.error(f"PostgreSQL insert failed: {e}")
            self.results.append(PerformanceMetrics(
                operation=f'{operation_type}_insert',
                database='postgresql',
                duration=time.time() - start_time,
                throughput=0,
                memory_usage=0,
                cpu_usage=0,
                records_processed=0,
                error_count=1,
                timestamp=datetime.now()
            ))
    
    async def test_postgresql_spatial_queries(self, conn, record_count: int):
        """Test PostgreSQL spatial query performance"""
        queries = [
            # Point-in-polygon query
            """
            SELECT COUNT(*) FROM seismic_events 
            WHERE ST_Within(location, ST_MakeEnvelope(-75, 40, -73, 41, 4326))
            """,
            
            # Distance-based query
            """
            SELECT event_id, magnitude, ST_Distance(location, ST_Point(-74.0060, 40.7128)) as distance
            FROM seismic_events 
            WHERE ST_DWithin(location, ST_Point(-74.0060, 40.7128), 0.1)
            ORDER BY distance LIMIT 100
            """,
            
            # Spatial clustering query
            """
            SELECT ST_ClusterKMeans(location, 10) OVER() as cluster_id, 
                   COUNT(*) as event_count,
                   AVG(magnitude) as avg_magnitude
            FROM seismic_events
            GROUP BY cluster_id
            """,
        ]
        
        for i, query in enumerate(queries):
            await self.time_query(conn, f'spatial_query_{i+1}', query, 'postgresql')
    
    async def test_postgresql_geological_queries(self, conn, record_count: int):
        """Test complex geological analysis queries"""
        queries = [
            # Magnitude distribution analysis
            """
            SELECT 
                FLOOR(magnitude) as mag_range,
                COUNT(*) as event_count,
                AVG(depth) as avg_depth,
                STDDEV(depth) as depth_stddev
            FROM seismic_events 
            GROUP BY FLOOR(magnitude)
            ORDER BY mag_range
            """,
            
            # Temporal clustering analysis
            """
            SELECT 
                DATE_TRUNC('hour', timestamp) as hour_bucket,
                COUNT(*) as events_per_hour,
                AVG(magnitude) as avg_magnitude,
                MAX(magnitude) as max_magnitude
            FROM seismic_events 
            GROUP BY hour_bucket
            HAVING COUNT(*) > 5
            ORDER BY hour_bucket
            """,
            
            # Depth-magnitude correlation
            """
            SELECT 
                CASE 
                    WHEN depth < 10 THEN 'shallow'
                    WHEN depth < 30 THEN 'intermediate'
                    ELSE 'deep'
                END as depth_category,
                COUNT(*) as event_count,
                AVG(magnitude) as avg_magnitude,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY magnitude) as p95_magnitude
            FROM seismic_events
            GROUP BY depth_category
            """,
        ]
        
        for i, query in enumerate(queries):
            await self.time_query(conn, f'geological_analysis_{i+1}', query, 'postgresql')
    
    async def test_influxdb_performance(self, record_count: int):
        """Test InfluxDB performance with time-series geological data"""
        logger.info(f"Testing InfluxDB with {record_count:,} records...")
        
        client = influxdb_client.InfluxDBClient(
            url=self.config['influxdb']['url'],
            token=self.config['influxdb']['token'],
            org=self.config['influxdb']['org']
        )
        
        try:
            write_api = client.write_api()
            query_api = client.query_api()
            
            # Test time-series data insertion
            await self.test_influxdb_insert(write_api, record_count)
            
            # Test time-series queries
            await self.test_influxdb_queries(query_api, record_count)
            
        finally:
            client.close()
    
    async def test_influxdb_insert(self, write_api, record_count: int):
        """Test InfluxDB insertion performance"""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        try:
            # Generate time-series seismic data
            points = []
            base_time = datetime.now() - timedelta(hours=record_count)
            
            for i in range(record_count):
                point = influxdb_client.Point("seismic_monitoring") \
                    .tag("station", f"station_{i % 20}") \
                    .tag("network", "prometheus") \
                    .field("magnitude", np.random.gamma(2, 0.8) + 1) \
                    .field("depth", np.random.exponential(15) + 1) \
                    .field("latitude", 40.7128 + np.random.normal(0, 0.1)) \
                    .field("longitude", -74.0060 + np.random.normal(0, 0.1)) \
                    .time(base_time + timedelta(seconds=i * 60))
                
                points.append(point)
                
                # Batch write every 1000 points
                if len(points) >= 1000:
                    write_api.write(bucket=self.config['influxdb']['bucket'], record=points)
                    points = []
            
            # Write remaining points
            if points:
                write_api.write(bucket=self.config['influxdb']['bucket'], record=points)
            
            duration = time.time() - start_time
            memory_used = psutil.virtual_memory().used - start_memory
            
            metrics = PerformanceMetrics(
                operation='timeseries_insert',
                database='influxdb',
                duration=duration,
                throughput=record_count / duration,
                memory_usage=memory_used,
                cpu_usage=psutil.cpu_percent(),
                records_processed=record_count,
                error_count=0,
                timestamp=datetime.now()
            )
            
            self.results.append(metrics)
            logger.info(f"InfluxDB insert: {record_count:,} records in {duration:.2f}s ({metrics.throughput:.0f} records/s)")
            
        except Exception as e:
            logger.error(f"InfluxDB insert failed: {e}")
    
    async def test_neo4j_performance(self, record_count: int):
        """Test Neo4j performance with geological relationship data"""
        logger.info(f"Testing Neo4j with {record_count:,} records...")
        
        driver = GraphDatabase.driver(
            self.config['neo4j']['uri'],
            auth=(self.config['neo4j']['user'], self.config['neo4j']['password'])
        )
        
        try:
            with driver.session() as session:
                # Test graph data insertion
                await self.test_neo4j_insert(session, record_count)
                
                # Test graph queries
                await self.test_neo4j_queries(session, record_count)
                
        finally:
            driver.close()
    
    async def test_redis_performance(self, record_count: int):
        """Test Redis performance with geological cache data"""
        logger.info(f"Testing Redis with {record_count:,} records...")
        
        r = redis.Redis(
            host=self.config['redis']['host'],
            port=self.config['redis']['port'],
            password=self.config['redis']['password'],
            decode_responses=True
        )
        
        try:
            # Test cache operations
            await self.test_redis_operations(r, record_count)
            
        finally:
            r.close()
    
    async def test_concurrent_operations(self, record_count: int):
        """Test concurrent database operations"""
        logger.info(f"Testing concurrent operations with {record_count:,} records...")
        
        # Simulate concurrent read/write operations
        tasks = []
        
        # Create multiple concurrent tasks
        for i in range(5):  # 5 concurrent operations
            task = asyncio.create_task(self.concurrent_operation_worker(i, record_count // 5))
            tasks.append(task)
        
        # Wait for all tasks to complete
        start_time = time.time()
        await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        metrics = PerformanceMetrics(
            operation='concurrent_operations',
            database='mixed',
            duration=duration,
            throughput=record_count / duration,
            memory_usage=psutil.virtual_memory().used,
            cpu_usage=psutil.cpu_percent(),
            records_processed=record_count,
            error_count=0,
            timestamp=datetime.now()
        )
        
        self.results.append(metrics)
        logger.info(f"Concurrent operations: {record_count:,} records in {duration:.2f}s")
    
    async def concurrent_operation_worker(self, worker_id: int, record_count: int):
        """Worker function for concurrent operations"""
        # Simulate mixed database operations
        await asyncio.sleep(0.1 * worker_id)  # Stagger start times
        
        # Perform database operations (simplified for testing)
        for i in range(record_count // 100):  # Batch operations
            await asyncio.sleep(0.01)  # Simulate database operation time
    
    async def time_query(self, conn, operation_name: str, query: str, database: str):
        """Time a database query and record metrics"""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        try:
            result = await conn.fetch(query)
            duration = time.time() - start_time
            memory_used = psutil.virtual_memory().used - start_memory
            
            metrics = PerformanceMetrics(
                operation=operation_name,
                database=database,
                duration=duration,
                throughput=len(result) / duration if result else 0,
                memory_usage=memory_used,
                cpu_usage=psutil.cpu_percent(),
                records_processed=len(result) if result else 0,
                error_count=0,
                timestamp=datetime.now()
            )
            
            self.results.append(metrics)
            logger.info(f"{operation_name}: {len(result) if result else 0} results in {duration:.3f}s")
            
        except Exception as e:
            logger.error(f"Query {operation_name} failed: {e}")
            self.results.append(PerformanceMetrics(
                operation=operation_name,
                database=database,
                duration=time.time() - start_time,
                throughput=0,
                memory_usage=0,
                cpu_usage=0,
                records_processed=0,
                error_count=1,
                timestamp=datetime.now()
            ))
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.results:
            return {"error": "No performance data collected"}
        
        # Convert results to DataFrame for analysis
        df = pd.DataFrame([
            {
                'operation': r.operation,
                'database': r.database,
                'duration': r.duration,
                'throughput': r.throughput,
                'memory_usage': r.memory_usage,
                'cpu_usage': r.cpu_usage,
                'records_processed': r.records_processed,
                'error_count': r.error_count,
                'timestamp': r.timestamp
            }
            for r in self.results
        ])
        
        # Calculate summary statistics
        summary = {
            'total_tests': len(self.results),
            'total_errors': df['error_count'].sum(),
            'average_throughput_by_db': df.groupby('database')['throughput'].mean().to_dict(),
            'average_duration_by_operation': df.groupby('operation')['duration'].mean().to_dict(),
            'memory_usage_stats': {
                'mean': df['memory_usage'].mean(),
                'max': df['memory_usage'].max(),
                'std': df['memory_usage'].std()
            },
            'performance_recommendations': self.generate_recommendations(df)
        }
        
        return summary
    
    def generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Check for slow operations
        slow_operations = df[df['duration'] > df['duration'].quantile(0.9)]
        if not slow_operations.empty:
            recommendations.append(
                f"Optimize slow operations: {', '.join(slow_operations['operation'].unique())}"
            )
        
        # Check for high memory usage
        high_memory = df[df['memory_usage'] > df['memory_usage'].quantile(0.9)]
        if not high_memory.empty:
            recommendations.append(
                "Consider memory optimization for high-usage operations"
            )
        
        # Check error rates
        error_rate = df['error_count'].sum() / len(df)
        if error_rate > 0.05:
            recommendations.append(
                f"High error rate detected ({error_rate:.1%}). Review error handling and database connections"
            )
        
        # Database-specific recommendations
        postgres_ops = df[df['database'] == 'postgresql']
        if not postgres_ops.empty and postgres_ops['throughput'].mean() < 1000:
            recommendations.append(
                "Consider PostgreSQL connection pooling and query optimization"
            )
        
        return recommendations
    
    def create_performance_visualizations(self):
        """Create performance visualization charts"""
        if not self.results:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'operation': r.operation,
                'database': r.database,
                'duration': r.duration,
                'throughput': r.throughput,
                'memory_usage': r.memory_usage / (1024**2),  # Convert to MB
                'records_processed': r.records_processed
            }
            for r in self.results
        ])
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Throughput by database
        sns.boxplot(data=df, x='database', y='throughput', ax=axes[0, 0])
        axes[0, 0].set_title('Throughput by Database')
        axes[0, 0].set_ylabel('Records/Second')
        
        # Duration by operation
        sns.barplot(data=df, x='operation', y='duration', ax=axes[0, 1])
        axes[0, 1].set_title('Duration by Operation')
        axes[0, 1].set_ylabel('Duration (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Memory usage distribution
        sns.histplot(data=df, x='memory_usage', bins=20, ax=axes[1, 0])
        axes[1, 0].set_title('Memory Usage Distribution')
        axes[1, 0].set_xlabel('Memory Usage (MB)')
        
        # Throughput vs Records Processed
        sns.scatterplot(data=df, x='records_processed', y='throughput', 
                       hue='database', ax=axes[1, 1])
        axes[1, 1].set_title('Throughput vs Records Processed')
        axes[1, 1].set_xlabel('Records Processed')
        axes[1, 1].set_ylabel('Throughput (records/s)')
        
        plt.tight_layout()
        plt.savefig('tests/performance/database-performance-report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Performance visualizations saved to database-performance-report.png")

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run database performance tests')
    parser.add_argument('--config', default='tests/performance/db-config.json',
                       help='Database configuration file')
    parser.add_argument('--output', default='tests/performance/db-performance-results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        # Default configuration
        config = {
            'postgres': {
                'host': 'localhost',
                'port': 5432,
                'user': 'prometheus',
                'password': 'prometheus123',
                'database': 'prometheus'
            },
            'influxdb': {
                'url': 'http://localhost:8086',
                'token': 'prometheus-token',
                'org': 'prometheus',
                'bucket': 'geological-data'
            },
            'neo4j': {
                'uri': 'bolt://localhost:7687',
                'user': 'neo4j',
                'password': 'prometheus123'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'password': None
            }
        }
    
    # Run performance tests
    async def main():
        tester = DatabasePerformanceTester(config)
        results = await tester.run_all_tests()
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Performance test results saved to {args.output}")
        print(f"Total tests: {results['total_tests']}")
        print(f"Total errors: {results['total_errors']}")
        
        if results['performance_recommendations']:
            print("\nRecommendations:")
            for rec in results['performance_recommendations']:
                print(f"  - {rec}")
    
    asyncio.run(main())