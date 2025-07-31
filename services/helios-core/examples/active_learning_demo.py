"""
Active Learning Engine Demonstration

This script demonstrates the capabilities of the active learning engine
for geological data selection and model improvement.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

# Add the core module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from active_learning_engine import (
    ActiveLearningEngine,
    ActiveLearningConfig,
    GeologicalSample,
    create_sample_geological_data_consistent
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_uncertainty_sampling():
    """Demonstrate uncertainty-based sampling strategy"""
    print("\n" + "="*60)
    print("UNCERTAINTY-BASED SAMPLING DEMONSTRATION")
    print("="*60)
    
    # Create synthetic geological dataset
    logger.info("Creating synthetic geological dataset...")
    all_samples = create_sample_geological_data_consistent(500)
    labeled_data = all_samples[:50]
    unlabeled_pool = all_samples[50:]
    
    # Configure uncertainty-based active learning
    config = ActiveLearningConfig(
        strategy="uncertainty",
        batch_size=15,
        uncertainty_threshold=0.1
    )
    
    engine = ActiveLearningEngine(config)
    
    # Create a simple model for demonstration
    import torch
    import torch.nn as nn
    
    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 1)
            self.dropout = nn.Dropout(0.3)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = DemoModel()
    
    # Select optimal samples
    logger.info("Selecting samples using uncertainty-based strategy...")
    result = engine.select_optimal_samples(
        unlabeled_pool=unlabeled_pool,
        labeled_data=labeled_data,
        model=model
    )
    
    # Display results
    print(f"\nSelected {len(result.selected_samples)} samples")
    print(f"Strategy used: {result.strategy_used}")
    print("\nEvaluation Metrics:")
    for metric, value in result.evaluation_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nGeological Coverage:")
    for coverage_type, value in result.geological_coverage.items():
        print(f"  {coverage_type}: {value:.4f}")
    
    # Analyze sample type distribution
    selected_types = [sample.sample_type for sample in result.selected_samples]
    type_counts = {}
    for sample_type in selected_types:
        type_counts[
class AdvancedGeologicalDatabase:
    """Advanced database operations for geological data with spatial indexing and time-series optimization"""
    
    def __init__(self, connection_params: Dict[str, str]):
        self.postgres_params = connection_params.get('postgres', {})
        self.influx_params = connection_params.get('influxdb', {})
        self.neo4j_params = connection_params.get('neo4j', {})
        self.redis_params = connection_params.get('redis', {})
        
        self.connections = {}
        self._initialize_connections()
        
    def _initialize_connections(self):
        """Initialize database connections with connection pooling"""
        import psycopg2.pool
        import influxdb_client
        from neo4j import GraphDatabase
        import redis
        
        # PostgreSQL connection pool
        if self.postgres_params:
            self.connections['postgres'] = psycopg2.pool.ThreadedConnectionPool(
                minconn=1, maxconn=20, **self.postgres_params
            )
        
        # InfluxDB client
        if self.influx_params:
            self.connections['influxdb'] = influxdb_client.InfluxDBClient(
                url=self.influx_params['url'],
                token=self.influx_params['token'],
                org=self.influx_params['org']
            )
        
        # Neo4j driver
        if self.neo4j_params:
            self.connections['neo4j'] = GraphDatabase.driver(
                self.neo4j_params['uri'],
                auth=(self.neo4j_params['user'], self.neo4j_params['password'])
            )
        
        # Redis connection
        if self.redis_params:
            self.connections['redis'] = redis.Redis(**self.redis_params)
    
    def store_seismic_data(self, seismic_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store seismic data with spatial and temporal indexing"""
        
        stored_count = 0
        errors = []
        
        # Get PostgreSQL connection
        postgres_conn = self.connections['postgres'].getconn()
        
        try:
            with postgres_conn.cursor() as cursor:
                # Create spatial index if not exists
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_seismic_location 
                    ON seismic_events USING GIST (location);
                    
                    CREATE INDEX IF NOT EXISTS idx_seismic_time 
                    ON seismic_events (event_time);
                    
                    CREATE INDEX IF NOT EXISTS idx_seismic_magnitude 
                    ON seismic_events (magnitude);
                """)
                
                # Prepare batch insert
                insert_query = """
                    INSERT INTO seismic_events 
                    (event_id, event_time, location, magnitude, depth, event_type, 
                     waveform_data, frequency_content, uncertainty_metrics)
                    VALUES (%s, %s, ST_GeomFromText(%s, 4326), %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO UPDATE SET
                        magnitude = EXCLUDED.magnitude,
                        depth = EXCLUDED.depth,
                        event_type = EXCLUDED.event_type,
                        waveform_data = EXCLUDED.waveform_data,
                        frequency_content = EXCLUDED.frequency_content,
                        uncertainty_metrics = EXCLUDED.uncertainty_metrics;
                """
                
                # Process events in batches
                batch_size = 1000
                for i in range(0, len(seismic_events), batch_size):
                    batch = seismic_events[i:i + batch_size]
                    batch_data = []
                    
                    for event in batch:
                        try:
                            # Convert location to WKT format
                            location_wkt = f"POINT({event['longitude']} {event['latitude']})"
                            
                            batch_data.append((
                                event['event_id'],
                                event['event_time'],
                                location_wkt,
                                event['magnitude'],
                                event['depth'],
                                event['event_type'],
                                json.dumps(event.get('waveform_data', {})),
                                json.dumps(event.get('frequency_content', {})),
                                json.dumps(event.get('uncertainty_metrics', {}))
                            ))
                            
                        except Exception as e:
                            errors.append(f"Event {event.get('event_id', 'unknown')}: {str(e)}")
                    
                    # Execute batch insert
                    cursor.executemany(insert_query, batch_data)
                    stored_count += len(batch_data)
                
                postgres_conn.commit()
                
        except Exception as e:
            postgres_conn.rollback()
            errors.append(f"Database error: {str(e)}")
        
        finally:
            self.connections['postgres'].putconn(postgres_conn)
        
        # Store time-series data in InfluxDB
        if 'influxdb' in self.connections:
            self._store_timeseries_data(seismic_events)
        
        return {
            'stored_count': stored_count,
            'total_events': len(seismic_events),
            'errors': errors,
            'success_rate': stored_count / len(seismic_events) if seismic_events else 0
        }
    
    def _store_timeseries_data(self, seismic_events: List[Dict[str, Any]]):
        """Store time-series data in InfluxDB"""
        from influxdb_client.client.write_api import SYNCHRONOUS
        
        write_api = self.connections['influxdb'].write_api(write_options=SYNCHRONOUS)
        
        points = []
        for event in seismic_events:
            # Create point for magnitude time series
            point = {
                "measurement": "seismic_magnitude",
                "tags": {
                    "event_type": event.get('event_type', 'unknown'),
                    "region": event.get('region', 'unknown'),
                    "station": event.get('station', 'unknown')
                },
                "fields": {
                    "magnitude": float(event.get('magnitude', 0)),
                    "depth": float(event.get('depth', 0)),
                    "latitude": float(event.get('latitude', 0)),
                    "longitude": float(event.get('longitude', 0))
                },
                "time": event.get('event_time')
            }
            points.append(point)
            
            # Store waveform data if available
            if 'waveform_data' in event:
                waveform_point = {
                    "measurement": "seismic_waveform",
                    "tags": {
                        "event_id": event.get('event_id'),
                        "component": "combined"
                    },
                    "fields": {
                        "amplitude": float(np.max(np.abs(event['waveform_data']))),
                        "rms": float(np.sqrt(np.mean(np.square(event['waveform_data'])))),
                        "peak_frequency": float(event.get('frequency_content', {}).get('dominant_frequency', 0))
                    },
                    "time": event.get('event_time')
                }
                points.append(waveform_point)
        
        try:
            write_api.write(bucket="geological_data", record=points)
        except Exception as e:
            logger.error(f"InfluxDB write error: {e}")
    
    def query_spatial_events(self, bbox: Tuple[float, float, float, float],
                           time_range: Tuple[str, str],
                           magnitude_range: Tuple[float, float] = (0, 10)) -> List[Dict[str, Any]]:
        """Query seismic events within spatial and temporal bounds"""
        
        min_lon, min_lat, max_lon, max_lat = bbox
        start_time, end_time = time_range
        min_mag, max_mag = magnitude_range
        
        postgres_conn = self.connections['postgres'].getconn()
        
        try:
            with postgres_conn.cursor() as cursor:
                query = """
                    SELECT event_id, event_time, ST_X(location) as longitude, 
                           ST_Y(location) as latitude, magnitude, depth, event_type,
                           waveform_data, frequency_content, uncertainty_metrics
                    FROM seismic_events
                    WHERE location && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
                      AND event_time BETWEEN %s AND %s
                      AND magnitude BETWEEN %s AND %s
                    ORDER BY event_time DESC
                    LIMIT 10000;
                """
                
                cursor.execute(query, (min_lon, min_lat, max_lon, max_lat, 
                                     start_time, end_time, min_mag, max_mag))
                
                results = []
                for row in cursor.fetchall():
                    event = {
                        'event_id': row[0],
                        'event_time': row[1],
                        'longitude': row[2],
                        'latitude': row[3],
                        'magnitude': row[4],
                        'depth': row[5],
                        'event_type': row[6],
                        'waveform_data': json.loads(row[7]) if row[7] else {},
                        'frequency_content': json.loads(row[8]) if row[8] else {},
                        'uncertainty_metrics': json.loads(row[9]) if row[9] else {}
                    }
                    results.append(event)
                
                return results
                
        except Exception as e:
            logger.error(f"Spatial query error: {e}")
            return []
        
        finally:
            self.connections['postgres'].putconn(postgres_conn)
    
    def create_geological_relationships(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create geological process relationships in Neo4j graph database"""
        
        if 'neo4j' not in self.connections:
            return {'error': 'Neo4j connection not available'}
        
        created_count = 0
        errors = []
        
        with self.connections['neo4j'].session() as session:
            try:
                # Create nodes and relationships
                for rel in relationships:
                    try:
                        # Create or update source node
                        session.run("""
                            MERGE (source:GeologicalProcess {id: $source_id})
                            SET source.name = $source_name,
                                source.type = $source_type,
                                source.properties = $source_props
                        """, 
                        source_id=rel['source_id'],
                        source_name=rel['source_name'],
                        source_type=rel['source_type'],
                        source_props=json.dumps(rel.get('source_properties', {})))
                        
                        # Create or update target node
                        session.run("""
                            MERGE (target:GeologicalProcess {id: $target_id})
                            SET target.name = $target_name,
                                target.type = $target_type,
                                target.properties = $target_props
                        """,
                        target_id=rel['target_id'],
                        target_name=rel['target_name'],
                        target_type=rel['target_type'],
                        target_props=json.dumps(rel.get('target_properties', {})))
                        
                        # Create relationship
                        session.run("""
                            MATCH (source:GeologicalProcess {id: $source_id})
                            MATCH (target:GeologicalProcess {id: $target_id})
                            MERGE (source)-[r:INFLUENCES {type: $rel_type}]->(target)
                            SET r.strength = $strength,
                                r.confidence = $confidence,
                                r.temporal_lag = $temporal_lag,
                                r.properties = $rel_props
                        """,
                        source_id=rel['source_id'],
                        target_id=rel['target_id'],
                        rel_type=rel['relationship_type'],
                        strength=rel.get('strength', 1.0),
                        confidence=rel.get('confidence', 0.5),
                        temporal_lag=rel.get('temporal_lag', 0),
                        rel_props=json.dumps(rel.get('relationship_properties', {})))
                        
                        created_count += 1
                        
                    except Exception as e:
                        errors.append(f"Relationship {rel.get('source_id', 'unknown')} -> {rel.get('target_id', 'unknown')}: {str(e)}")
                
            except Exception as e:
                errors.append(f"Neo4j session error: {str(e)}")
        
        return {
            'created_count': created_count,
            'total_relationships': len(relationships),
            'errors': errors,
            'success_rate': created_count / len(relationships) if relationships else 0
        }
    
    def cache_analysis_results(self, analysis_id: str, results: Dict[str, Any], 
                             expiration_hours: int = 24) -> bool:
        """Cache analysis results in Redis with expiration"""
        
        if 'redis' not in self.connections:
            return False
        
        try:
            redis_client = self.connections['redis']
            
            # Store results with expiration
            cache_key = f"analysis:{analysis_id}"
            redis_client.setex(
                cache_key,
                expiration_hours * 3600,
                json.dumps(results, default=str)
            )
            
            # Store metadata
            metadata_key = f"analysis_meta:{analysis_id}"
            metadata = {
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=expiration_hours)).isoformat(),
                'size_bytes': len(json.dumps(results, default=str))
            }
            redis_client.setex(
                metadata_key,
                expiration_hours * 3600,
                json.dumps(metadata)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Redis cache error: {e}")
            return False
    
    def get_cached_results(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis results"""
        
        if 'redis' not in self.connections:
            return None
        
        try:
            redis_client = self.connections['redis']
            cache_key = f"analysis:{analysis_id}"
            
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Redis retrieval error: {e}")
            return None
 type_counts.get(sample_type, 0) + 1
    
    print(f"\nSelected Sample Types Distribution:")
    for sample_type, count in type_counts.items():
        print(f"  {sample_type}: {count} samples")
    
    return result


def demonstrate_diversity_sampling():
    """Demonstrate diversity-based sampling strategy"""
    print("\n" + "="*60)
    print("DIVERSITY-BASED SAMPLING DEMONSTRATION")
    print("="*60)
    
    # Create synthetic geological dataset
    logger.info("Creating synthetic geological dataset...")
    all_samples = create_sample_geological_data_consistent(500)
    labeled_data = all_samples[:50]
    unlabeled_pool = all_samples[50:]
    
    # Configure diversity-based active learning
    config = ActiveLearningConfig(
        strategy="diversity",
        batch_size=15,
        diversity_weight=0.8
    )
    
    engine = ActiveLearningEngine(config)
    
    # Create a simple model for demonstration
    import torch
    import torch.nn as nn
    
    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 1)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = DemoModel()
    
    # Select optimal samples
    logger.info("Selecting samples using diversity-based strategy...")
    result = engine.select_optimal_samples(
        unlabeled_pool=unlabeled_pool,
        labeled_data=labeled_data,
        model=model
    )
    
    # Display results
    print(f"\nSelected {len(result.selected_samples)} samples")
    print(f"Strategy used: {result.strategy_used}")
    print("\nEvaluation Metrics:")
    for metric, value in result.evaluation_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nGeological Coverage:")
    for coverage_type, value in result.geological_coverage.items():
        print(f"  {coverage_type}: {value:.4f}")
    
    # Analyze geographic distribution
    locations = np.array([sample.location[:2] for sample in result.selected_samples])
    print(f"\nGeographic Distribution:")
    print(f"  Latitude range: {np.min(locations[:, 0]):.2f} to {np.max(locations[:, 0]):.2f}")
    print(f"  Longitude range: {np.min(locations[:, 1]):.2f} to {np.max(locations[:, 1]):.2f}")
    print(f"  Geographic spread: {np.std(locations.flatten()):.2f}")
    
    return result


def demonstrate_hybrid_sampling():
    """Demonstrate hybrid sampling strategy"""
    print("\n" + "="*60)
    print("HYBRID SAMPLING DEMONSTRATION")
    print("="*60)
    
    # Create synthetic geological dataset
    logger.info("Creating synthetic geological dataset...")
    all_samples = create_sample_geological_data_consistent(500)
    labeled_data = all_samples[:50]
    unlabeled_pool = all_samples[50:]
    
    # Configure hybrid active learning
    config = ActiveLearningConfig(
        strategy="hybrid",
        batch_size=15,
        diversity_weight=0.4,  # Balance between uncertainty and diversity
        uncertainty_threshold=0.05
    )
    
    engine = ActiveLearningEngine(config)
    
    # Create a simple model for demonstration
    import torch
    import torch.nn as nn
    
    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = DemoModel()
    
    # Select optimal samples
    logger.info("Selecting samples using hybrid strategy...")
    result = engine.select_optimal_samples(
        unlabeled_pool=unlabeled_pool,
        labeled_data=labeled_data,
        model=model
    )
    
    # Display results
    print(f"\nSelected {len(result.selected_samples)} samples")
    print(f"Strategy used: {result.strategy_used}")
    print("\nEvaluation Metrics:")
    for metric, value in result.evaluation_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nGeological Coverage:")
    for coverage_type, value in result.geological_coverage.items():
        print(f"  {coverage_type}: {value:.4f}")
    
    return result


def demonstrate_strategy_comparison():
    """Compare different active learning strategies"""
    print("\n" + "="*60)
    print("STRATEGY COMPARISON DEMONSTRATION")
    print("="*60)
    
    # Create synthetic geological dataset
    logger.info("Creating synthetic geological dataset for comparison...")
    all_samples = create_sample_geological_data_consistent(1000)
    labeled_data = all_samples[:100]
    unlabeled_pool = all_samples[100:]
    
    # Create model
    import torch
    import torch.nn as nn
    
    class ComparisonModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = ComparisonModel()
    
    # Test different strategies
    strategies = ['uncertainty', 'diversity', 'hybrid']
    results = {}
    
    for strategy in strategies:
        logger.info(f"Testing {strategy} strategy...")
        
        config = ActiveLearningConfig(
            strategy=strategy,
            batch_size=25,
            diversity_weight=0.3
        )
        
        engine = ActiveLearningEngine(config)
        
        result = engine.select_optimal_samples(
            unlabeled_pool=unlabeled_pool,
            labeled_data=labeled_data,
            model=model
        )
        
        results[strategy] = result
    
    # Compare results
    print("\nSTRATEGY COMPARISON RESULTS:")
    print("-" * 40)
    
    metrics_to_compare = [
        'sample_type_diversity',
        'diversity_score',
        'mean_uncertainty',
        'geological_information_gain'
    ]
    
    for metric in metrics_to_compare:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for strategy, result in results.items():
            value = result.evaluation_metrics.get(metric, 0.0)
            print(f"  {strategy:12}: {value:.4f}")
    
    # Coverage comparison
    print(f"\nGeological Coverage Comparison:")
    coverage_metrics = ['sample_type_coverage', 'geographic_coverage', 'depth_coverage']
    
    for metric in coverage_metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for strategy, result in results.items():
            value = result.geological_coverage.get(metric, 0.0)
            if metric == 'geographic_coverage':
                print(f"  {strategy:12}: {value:.2f}")
            else:
                print(f"  {strategy:12}: {value:.4f}")
    
    # Recommend best strategy
    print(f"\nSTRATEGY RECOMMENDATIONS:")
    print("-" * 30)
    
    recommendations = engine.get_strategy_recommendations(unlabeled_pool, labeled_data)
    for key, value in recommendations.items():
        print(f"{key}: {value}")
    
    return results


def demonstrate_iterative_learning():
    """Demonstrate iterative active learning process"""
    print("\n" + "="*60)
    print("ITERATIVE ACTIVE LEARNING DEMONSTRATION")
    print("="*60)
    
    # Create synthetic geological dataset
    logger.info("Creating synthetic geological dataset for iterative learning...")
    all_samples = create_sample_geological_data_consistent(1000)
    labeled_data = all_samples[:50]
    unlabeled_pool = all_samples[50:]
    
    # Create model
    import torch
    import torch.nn as nn
    
    class IterativeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = IterativeModel()
    
    # Configure active learning
    config = ActiveLearningConfig(
        strategy="hybrid",
        batch_size=20,
        diversity_weight=0.3
    )
    
    engine = ActiveLearningEngine(config)
    
    # Simulate iterative learning process
    iterations = 5
    iteration_results = []
    
    print(f"Starting iterative learning with {len(labeled_data)} labeled samples")
    print(f"Unlabeled pool size: {len(unlabeled_pool)}")
    
    for iteration in range(iterations):
        logger.info(f"Iteration {iteration + 1}/{iterations}")
        
        # Select optimal samples
        result = engine.select_optimal_samples(
            unlabeled_pool=unlabeled_pool,
            labeled_data=labeled_data,
            model=model
        )
        
        # Move selected samples to labeled data
        selected_samples = result.selected_samples
        labeled_data.extend(selected_samples)
        
        # Remove selected samples from unlabeled pool
        selected_ids = {sample.id for sample in selected_samples}
        unlabeled_pool = [sample for sample in unlabeled_pool 
                         if sample.id not in selected_ids]
        
        # Store iteration results
        iteration_results.append({
            'iteration': iteration + 1,
            'labeled_count': len(labeled_data),
            'unlabeled_count': len(unlabeled_pool),
            'selected_count': len(selected_samples),
            'diversity_score': result.evaluation_metrics.get('diversity_score', 0.0),
            'sample_type_diversity': result.evaluation_metrics.get('sample_type_diversity', 0.0),
            'geological_coverage': result.geological_coverage.get('sample_type_coverage', 0.0)
        })
        
        print(f"  Iteration {iteration + 1}: Selected {len(selected_samples)} samples")
        print(f"    Labeled data: {len(labeled_data)} samples")
        print(f"    Unlabeled pool: {len(unlabeled_pool)} samples")
        print(f"    Diversity score: {result.evaluation_metrics.get('diversity_score', 0.0):.4f}")
    
    # Display final results
    print(f"\nITERATIVE LEARNING SUMMARY:")
    print("-" * 30)
    print(f"Initial labeled samples: 50")
    print(f"Final labeled samples: {len(labeled_data)}")
    print(f"Total samples selected: {len(labeled_data) - 50}")
    print(f"Remaining unlabeled samples: {len(unlabeled_pool)}")
    
    # Show progression
    print(f"\nLearning Progression:")
    for result in iteration_results:
        print(f"  Iteration {result['iteration']}: "
              f"Labeled={result['labeled_count']}, "
              f"Diversity={result['diversity_score']:.3f}, "
              f"Coverage={result['geological_coverage']:.3f}")
    
    return iteration_results


def main():
    """Main demonstration function"""
    print("ACTIVE LEARNING ENGINE DEMONSTRATION")
    print("=" * 80)
    print("This demonstration showcases the capabilities of the active learning engine")
    print("for optimal geological data selection and model improvement.")
    
    try:
        # Run demonstrations
        uncertainty_result = demonstrate_uncertainty_sampling()
        diversity_result = demonstrate_diversity_sampling()
        hybrid_result = demonstrate_hybrid_sampling()
        comparison_results = demonstrate_strategy_comparison()
        iterative_results = demonstrate_iterative_learning()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print("All active learning strategies have been demonstrated.")
        print("The engine is ready for integration with geological modeling workflows.")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        print(f"\nError during demonstration: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\nActive learning engine demonstration completed successfully!")
    else:
        print("\nActive learning engine demonstration failed!")
        sys.exit(1)