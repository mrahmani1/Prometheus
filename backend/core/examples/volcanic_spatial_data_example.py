"""
Volcanic Spatial Data Connector Usage Example

This example demonstrates how to use the volcanic spatial data connector
to retrieve real volcanic data for emergent behavior analysis.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Basic usage example of the volcanic spatial data connector"""
    
    try:
        from backend.core.volcanic_spatial_data_connector import VolcanicSpatialDataConnector
        from backend.core.database_integration import DatabaseIntegration
        
        logger.info("=== Basic Usage Example ===")
        
        # Initialize database and connector
        db_integration = DatabaseIntegration()
        await db_integration.initialize()
        
        connector = VolcanicSpatialDataConnector(db_integration)
        await connector.initialize()
        
        # Get global volcanic data
        logger.info("Fetching global volcanic spatial data...")
        global_data = await connector.get_real_volcanic_spatial_data()
        
        if global_data is not None:
            logger.info(f"Retrieved {len(global_data)} volcanic data points")
            logger.info(f"Data columns: [latitude, longitude, elevation, activity_score]")
            logger.info(f"Sample point: {global_data[0]}")
            
            # Basic statistics
            logger.info(f"Latitude range: {global_data[:, 0].min():.2f} to {global_data[:, 0].max():.2f}")
            logger.info(f"Longitude range: {global_data[:, 1].min():.2f} to {global_data[:, 1].max():.2f}")
            logger.info(f"Elevation range: {global_data[:, 2].min():.0f} to {global_data[:, 2].max():.0f} meters")
            logger.info(f"Activity score range: {global_data[:, 3].min():.3f} to {global_data[:, 3].max():.3f}")
        else:
            logger.info("No real data available - would use synthetic data in production")
        
        # Cleanup
        await connector.shutdown()
        await db_integration.close()
        
    except Exception as e:
        logger.error(f"Basic usage example failed: {e}")


async def example_regional_analysis():
    """Example of regional volcanic analysis"""
    
    try:
        from backend.core.volcanic_spatial_data_connector import VolcanicSpatialDataConnector
        from backend.core.database_integration import DatabaseIntegration
        
        logger.info("=== Regional Analysis Example ===")
        
        # Initialize
        db_integration = DatabaseIntegration()
        await db_integration.initialize()
        
        connector = VolcanicSpatialDataConnector(db_integration)
        await connector.initialize()
        
        # Define regions of interest
        regions = {
            "Cascade Range": (42, -125, 49, -120),
            "Kamchatka Peninsula": (50, 155, 65, 165),
            "Iceland": (63, -25, 67, -13),
            "Java, Indonesia": (-9, 105, -6, 115),
            "Andes (Central Chile)": (-40, -75, -30, -68)
        }
        
        for region_name, bounds in regions.items():
            logger.info(f"\nAnalyzing {region_name}...")
            
            # Get regional data
            regional_data = await connector.get_real_volcanic_spatial_data(
                region_bounds=bounds,
                min_activity_level='green',
                include_historical=True
            )
            
            if regional_data is not None and len(regional_data) > 0:
                logger.info(f"  Found {len(regional_data)} volcanoes")
                
                # Calculate regional statistics
                avg_elevation = np.mean(regional_data[:, 2])
                avg_activity = np.mean(regional_data[:, 3])
                
                logger.info(f"  Average elevation: {avg_elevation:.0f} meters")
                logger.info(f"  Average activity score: {avg_activity:.3f}")
                
                # Get clusters for this region
                clusters = await connector.get_volcanic_clusters(bounds)
                if clusters:
                    logger.info(f"  Identified {len(clusters)} volcanic clusters")
                    for i, cluster in enumerate(clusters):
                        logger.info(f"    Cluster {i+1}: {cluster.volcano_count} volcanoes, "
                                  f"radius {cluster.radius_km:.1f}km")
                else:
                    logger.info("  No distinct clusters identified")
            else:
                logger.info(f"  No volcanic data found for {region_name}")
        
        # Cleanup
        await connector.shutdown()
        await db_integration.close()
        
    except Exception as e:
        logger.error(f"Regional analysis example failed: {e}")


async def example_activity_filtering():
    """Example of filtering volcanoes by activity level"""
    
    try:
        from backend.core.volcanic_spatial_data_connector import VolcanicSpatialDataConnector
        from backend.core.database_integration import DatabaseIntegration
        
        logger.info("=== Activity Level Filtering Example ===")
        
        # Initialize
        db_integration = DatabaseIntegration()
        await db_integration.initialize()
        
        connector = VolcanicSpatialDataConnector(db_integration)
        await connector.initialize()
        
        # Test different activity level filters
        activity_levels = ['green', 'yellow', 'orange', 'red']
        
        # Use Pacific Ring of Fire as test region
        pacific_bounds = (10, 120, 60, -120)
        
        for level in activity_levels:
            logger.info(f"\nFetching volcanoes with minimum activity level: {level}")
            
            data = await connector.get_real_volcanic_spatial_data(
                region_bounds=pacific_bounds,
                min_activity_level=level,
                include_historical=False  # Only currently active
            )
            
            if data is not None:
                count = len(data)
                high_activity_count = np.sum(data[:, 3] > 0.5)
                
                logger.info(f"  Total volcanoes: {count}")
                logger.info(f"  High activity (>0.5): {high_activity_count}")
                
                if count > 0:
                    logger.info(f"  Activity score range: {data[:, 3].min():.3f} - {data[:, 3].max():.3f}")
            else:
                logger.info(f"  No data available for activity level {level}")
        
        # Cleanup
        await connector.shutdown()
        await db_integration.close()
        
    except Exception as e:
        logger.error(f"Activity filtering example failed: {e}")


async def example_cluster_analysis():
    """Example of volcanic cluster analysis with emergent properties"""
    
    try:
        from backend.core.volcanic_spatial_data_connector import VolcanicSpatialDataConnector
        from backend.core.database_integration import DatabaseIntegration
        
        logger.info("=== Cluster Analysis Example ===")
        
        # Initialize
        db_integration = DatabaseIntegration()
        await db_integration.initialize()
        
        connector = VolcanicSpatialDataConnector(db_integration)
        await connector.initialize()
        
        # Analyze clusters in different regions
        test_regions = {
            "Indonesia": (-10, 95, 6, 141),
            "Japan": (30, 129, 46, 146),
            "Alaska": (54, -170, 71, -130),
            "Italy": (36, 6, 47, 19)
        }
        
        for region_name, bounds in test_regions.items():
            logger.info(f"\nAnalyzing volcanic clusters in {region_name}...")
            
            clusters = await connector.get_volcanic_clusters(bounds)
            
            if clusters:
                logger.info(f"Found {len(clusters)} clusters:")
                
                for i, cluster in enumerate(clusters):
                    logger.info(f"\n  Cluster {i+1} ({cluster.cluster_id}):")
                    logger.info(f"    Location: {cluster.center_lat:.2f}°N, {cluster.center_lon:.2f}°E")
                    logger.info(f"    Volcanoes: {cluster.volcano_count}")
                    logger.info(f"    Radius: {cluster.radius_km:.1f} km")
                    logger.info(f"    Activity Level: {cluster.activity_level}")
                    logger.info(f"    Dominant Type: {cluster.dominant_type}")
                    logger.info(f"    Tectonic Setting: {cluster.tectonic_setting}")
                    
                    # Show emergent properties
                    props = cluster.emergent_properties
                    logger.info(f"    Emergent Properties:")
                    logger.info(f"      Spatial Density: {props.get('spatial_density', 0):.4f} volcanoes/km²")
                    logger.info(f"      Elevation Variance: {props.get('elevation_variance', 0):.0f} m²")
                    logger.info(f"      Activity Correlation: {props.get('activity_correlation', 0):.3f}")
                    logger.info(f"      Fractal Dimension: {props.get('fractal_dimension', 0):.3f}")
            else:
                logger.info(f"No clusters found in {region_name}")
        
        # Cleanup
        await connector.shutdown()
        await db_integration.close()
        
    except Exception as e:
        logger.error(f"Cluster analysis example failed: {e}")


async def example_integration_with_emergent_behavior():
    """Example showing integration with emergent behavior detection"""
    
    try:
        logger.info("=== Integration with Emergent Behavior Detection ===")
        
        # Simulate the emergent behavior detection system usage
        from backend.core.volcanic_spatial_data_connector import VolcanicSpatialDataConnector
        from backend.core.database_integration import DatabaseIntegration
        
        # Initialize components
        db_integration = DatabaseIntegration()
        await db_integration.initialize()
        
        connector = VolcanicSpatialDataConnector(db_integration)
        await connector.initialize()
        
        # Simulate the emergent behavior system's data request
        logger.info("Simulating emergent behavior system data request...")
        
        spatial_data = await connector.get_real_volcanic_spatial_data(
            region_bounds=None,  # Global analysis
            min_activity_level='green',
            include_historical=True
        )
        
        if spatial_data is not None and len(spatial_data) > 0:
            logger.info(f"✓ Real volcanic data available: {len(spatial_data)} points")
            
            # Simulate emergent behavior analysis
            logger.info("Performing emergent behavior analysis on real data...")
            
            # Calculate some basic emergent properties
            coords = spatial_data[:, :2]  # lat, lon
            elevations = spatial_data[:, 2]
            activities = spatial_data[:, 3]
            
            # Spatial clustering tendency
            from scipy.spatial.distance import pdist
            if len(coords) > 1:
                distances = pdist(coords)
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                clustering_coefficient = std_distance / mean_distance if mean_distance > 0 else 0
                
                logger.info(f"  Mean inter-volcano distance: {mean_distance:.2f}°")
                logger.info(f"  Clustering coefficient: {clustering_coefficient:.3f}")
            
            # Elevation-activity correlation
            if len(elevations) > 1 and len(activities) > 1:
                correlation = np.corrcoef(elevations, activities)[0, 1]
                logger.info(f"  Elevation-activity correlation: {correlation:.3f}")
            
            # Activity distribution
            high_activity = np.sum(activities > 0.5)
            moderate_activity = np.sum((activities > 0.2) & (activities <= 0.5))
            low_activity = np.sum(activities <= 0.2)
            
            logger.info(f"  Activity distribution:")
            logger.info(f"    High activity (>0.5): {high_activity}")
            logger.info(f"    Moderate activity (0.2-0.5): {moderate_activity}")
            logger.info(f"    Low activity (≤0.2): {low_activity}")
            
        else:
            logger.info("⚠ No real volcanic data available")
            logger.info("System would fall back to synthetic data generation")
            
            # Demonstrate synthetic data generation
            logger.info("Generating synthetic volcanic data...")
            
            # Create a simple synthetic dataset
            np.random.seed(42)  # For reproducible results
            n_volcanoes = 50
            
            # Generate realistic volcanic locations (concentrated in known regions)
            synthetic_data = []
            
            # Pacific Ring of Fire volcanoes
            for _ in range(30):
                lat = np.random.uniform(-60, 70)
                lon = np.random.uniform(120, -60) if np.random.random() > 0.5 else np.random.uniform(-180, -60)
                elevation = np.random.uniform(0, 4000)
                activity = np.random.exponential(0.2)  # Most volcanoes have low activity
                synthetic_data.append([lat, lon, elevation, min(activity, 1.0)])
            
            # Mid-Atlantic Ridge volcanoes
            for _ in range(10):
                lat = np.random.uniform(-60, 80)
                lon = np.random.uniform(-40, 20)
                elevation = np.random.uniform(-2000, 2000)  # Some submarine
                activity = np.random.exponential(0.15)
                synthetic_data.append([lat, lon, elevation, min(activity, 1.0)])
            
            # Mediterranean volcanoes
            for _ in range(10):
                lat = np.random.uniform(35, 45)
                lon = np.random.uniform(5, 25)
                elevation = np.random.uniform(0, 3500)
                activity = np.random.exponential(0.25)
                synthetic_data.append([lat, lon, elevation, min(activity, 1.0)])
            
            synthetic_array = np.array(synthetic_data)
            logger.info(f"✓ Generated {len(synthetic_array)} synthetic volcanic data points")
            logger.info(f"  Synthetic data shape: {synthetic_array.shape}")
            logger.info(f"  Sample synthetic point: {synthetic_array[0]}")
        
        # Cleanup
        await connector.shutdown()
        await db_integration.close()
        
        logger.info("✓ Integration example completed successfully")
        
    except Exception as e:
        logger.error(f"Integration example failed: {e}")


async def main():
    """Run all usage examples"""
    
    logger.info("🌋 VOLCANIC SPATIAL DATA CONNECTOR EXAMPLES")
    logger.info("=" * 60)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Regional Analysis", example_regional_analysis),
        ("Activity Filtering", example_activity_filtering),
        ("Cluster Analysis", example_cluster_analysis),
        ("Emergent Behavior Integration", example_integration_with_emergent_behavior)
    ]
    
    for name, example_func in examples:
        try:
            logger.info(f"\n{'='*20} {name} {'='*20}")
            await example_func()
            logger.info(f"✓ {name} completed successfully")
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
        
        # Small delay between examples
        await asyncio.sleep(1)
    
    logger.info("\n" + "="*60)
    logger.info("🎉 All examples completed!")
    logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())