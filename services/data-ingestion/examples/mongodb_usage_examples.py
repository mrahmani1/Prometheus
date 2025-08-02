"""Examples demonstrating MongoDB storage usage for geological data"""

import asyncio
from datetime import datetime, timedelta

import torch.nn as nn
import torch.nn.functional as F
from models.geological_data import (AtmosphericMeasurement, DataQuality,
                                    GeochemicalSample, GeologicalLocation)
from storage.mongodb_storage import MongoDBStorage


async def example_basic_storage():
    """Example: Basic storage operations"""
    print("=== Basic Storage Operations ===")

    # Initialize MongoDB storage
    storage = MongoDBStorage("mongodb://localhost:27017", "prometheus_examples")
    await storage.initialize()

    try:
        # Create sample location
        location = GeologicalLocation(
            latitude=45.3311,
            longitude=-121.7113,  # Mount Hood, Oregon
            elevation=3429.0,
            depth=0.0,
            coordinate_system="WGS84",
        )

        # Store geochemical sample
        geochemical_sample = GeochemicalSample(
            timestamp=datetime.utcnow(),
            location=location,
            sample_type="volcanic_rock",
            composition={
                "SiO2": 58.2,
                "Al2O3": 17.1,
                "FeO": 8.9,
                "MgO": 4.2,
                "CaO": 7.8,
                "Na2O": 2.9,
                "K2O": 0.9,
            },
            isotope_ratios={
                "Sr87_Sr86": 0.7038,
                "Nd143_Nd144": 0.5129,
                "Pb206_Pb204": 18.95,
            },
            temperature=1150.0,
            pressure=2.1e8,
            ph=6.8,
            source="USGS_Mount_Hood_Survey",
            quality=DataQuality.EXCELLENT,
        )

        sample_id = await storage.store_geochemical_sample(geochemical_sample)
        print(f"Stored geochemical sample: {sample_id}")

        # Store atmospheric measurement
        atmospheric_measurement = AtmosphericMeasurement(
            timestamp=datetime.utcnow(),
            location=location,
            pressure=70000.0,  # Pa (high altitude)
            temperature=268.15,  # K (-5°C)
            humidity=85.0,
            wind_speed=12.5,
            wind_direction=270.0,
            gas_concentrations={"CO2": 415.0, "SO2": 0.12, "H2S": 0.03, "CH4": 1.9},
            source="Mount_Hood_Weather_Station",
            quality=DataQuality.GOOD,
        )

        measurement_id = await storage.store_atmospheric_measurement(
            atmospheric_measurement
        )
        print(f"Stored atmospheric measurement: {measurement_id}")

    finally:
        await storage.cleanup()


async def example_unstructured_data():
    """Example: Storing various unstructured geological data types"""
    print("\n=== Unstructured Data Storage ===")

    storage = MongoDBStorage("mongodb://localhost:27017", "prometheus_examples")
    await storage.initialize()

    try:
        # Rock sample data
        rock_sample_data = {
            "timestamp": datetime.utcnow(),
            "location": {
                "latitude": 44.0582,
                "longitude": -121.3153,  # Crater Lake, Oregon
                "elevation": 1883.0,
            },
            "sample_id": "CL_2024_001",
            "rock_type": "volcanic",
            "formation": "Mount Mazama",
            "age_ma": 0.0077,  # ~7,700 years ago
            "texture": "vesicular",
            "grain_size": "fine",
            "color": "dark_gray",
            "mineral_assemblage": ["plagioclase", "pyroxene", "olivine", "magnetite"],
            "alteration_type": "none",
            "field_notes": "Pumice sample from Crater Lake caldera rim",
            "collector": "Dr. Jane Smith",
            "collection_date": datetime.utcnow(),
            "photos": ["CL_2024_001_field.jpg", "CL_2024_001_hand_sample.jpg"],
            "source": "USGS_Crater_Lake_Study",
            "quality": "excellent",
        }

        rock_id = await storage.store_unstructured_data(
            rock_sample_data, "rock_samples"
        )
        print(f"Stored rock sample: {rock_id}")

        # Mineral analysis data
        mineral_analysis_data = {
            "timestamp": datetime.utcnow(),
            "sample_id": "CL_2024_001_MIN_01",
            "mineral_name": "plagioclase",
            "chemical_formula": "(Ca,Na)(Al,Si)4O8",
            "crystal_system": "triclinic",
            "hardness": 6.5,
            "specific_gravity": 2.65,
            "color": "white_to_gray",
            "optical_properties": {
                "refractive_index": 1.56,
                "birefringence": 0.007,
                "extinction_angle": 15.0,
            },
            "chemical_composition": {
                "SiO2": 52.1,
                "Al2O3": 30.2,
                "CaO": 12.8,
                "Na2O": 4.9,
            },
            "analysis_method": "electron_microprobe",
            "analyst": "Dr. Mike Johnson",
            "laboratory": "Oregon_State_University",
            "analysis_date": datetime.utcnow(),
            "source": "OSU_Mineral_Lab",
            "quality": "excellent",
        }

        mineral_id = await storage.store_unstructured_data(
            mineral_analysis_data, "mineral_analyses"
        )
        print(f"Stored mineral analysis: {mineral_id}")

        # Field observation data
        field_observation_data = {
            "timestamp": datetime.utcnow(),
            "location": {
                "latitude": 45.3311,
                "longitude": -121.7113,
                "elevation": 3200.0,
            },
            "observation_id": "MH_2024_OBS_001",
            "observer": "Dr. Sarah Wilson",
            "observation_type": "outcrop",
            "description": "Well-exposed andesite flow with columnar jointing",
            "geological_unit": "Mount_Hood_Andesite",
            "rock_type": "igneous",
            "strike": 045.0,
            "dip": 25.0,
            "thickness": 15.0,
            "color": "medium_gray",
            "texture": "porphyritic",
            "weathering": "moderate",
            "accessibility": "moderate_hike",
            "confidence_level": "high",
            "photos": ["MH_OBS_001_overview.jpg", "MH_OBS_001_detail.jpg"],
            "gps_waypoints": [
                {"latitude": 45.3311, "longitude": -121.7113, "elevation": 3200.0},
                {"latitude": 45.3315, "longitude": -121.7110, "elevation": 3205.0},
            ],
            "source": "Mount_Hood_Field_Survey_2024",
            "quality": "good",
        }

        obs_id = await storage.store_unstructured_data(
            field_observation_data, "field_observations"
        )
        print(f"Stored field observation: {obs_id}")

    finally:
        await storage.cleanup()


async def example_bulk_operations():
    """Example: Bulk data operations"""
    print("\n=== Bulk Operations ===")

    storage = MongoDBStorage("mongodb://localhost:27017", "prometheus_examples")
    await storage.initialize()

    try:
        # Generate bulk geochemical data
        bulk_samples = []
        base_location = {"latitude": 44.0, "longitude": -121.0, "elevation": 2000.0}

        for i in range(10):
            sample = {
                "timestamp": datetime.utcnow() - timedelta(days=i),
                "location": {
                    "latitude": base_location["latitude"] + i * 0.01,
                    "longitude": base_location["longitude"] + i * 0.01,
                    "elevation": base_location["elevation"] + i * 10,
                },
                "sample_id": f"BULK_SAMPLE_{i:03d}",
                "sample_type": "volcanic_rock",
                "composition": {
                    "SiO2": 55.0 + i * 2.0,
                    "Al2O3": 16.0 + i * 0.5,
                    "FeO": 9.0 - i * 0.3,
                    "MgO": 5.0 - i * 0.2,
                    "CaO": 8.0 - i * 0.4,
                },
                "temperature": 1100.0 + i * 20.0,
                "source": "Bulk_Survey_2024",
                "quality": "good",
            }
            bulk_samples.append(sample)

        # Bulk insert
        inserted_ids = await storage.bulk_insert(bulk_samples, "geochemical_samples")
        print(f"Bulk inserted {len(inserted_ids)} geochemical samples")

    finally:
        await storage.cleanup()


async def example_advanced_queries():
    """Example: Advanced querying capabilities"""
    print("\n=== Advanced Queries ===")

    storage = MongoDBStorage("mongodb://localhost:27017", "prometheus_examples")
    await storage.initialize()

    try:
        # First, store some test data
        await example_bulk_operations()

        # Location-based query
        print("\n--- Location-based Query ---")
        location_results = await storage.query_by_location(
            longitude=-121.0,
            latitude=44.0,
            radius_km=50.0,
            collection_name="geochemical_samples",
            limit=5,
        )
        print(f"Found {len(location_results)} samples within 50km of target location")
        for result in location_results[:2]:  # Show first 2
            print(
                f"  Sample {result.get('sample_id', 'N/A')}: "
                f"SiO2={result.get('composition', {}).get('SiO2', 'N/A')}%"
            )

        # Time range query
        print("\n--- Time Range Query ---")
        start_time = datetime.utcnow() - timedelta(days=5)
        end_time = datetime.utcnow()

        time_results = await storage.query_by_time_range(
            start_time=start_time,
            end_time=end_time,
            collection_name="geochemical_samples",
            limit=5,
        )
        print(f"Found {len(time_results)} samples in the last 5 days")

        # Text search
        print("\n--- Text Search ---")
        search_results = await storage.text_search(
            "volcanic rock", "geochemical_samples", limit=3
        )
        print(f"Found {len(search_results)} samples matching 'volcanic rock'")

    finally:
        await storage.cleanup()


async def example_aggregation_pipelines():
    """Example: Advanced aggregation pipelines"""
    print("\n=== Aggregation Pipelines ===")

    storage = MongoDBStorage("mongodb://localhost:27017", "prometheus_examples")
    await storage.initialize()

    try:
        # Ensure we have data
        await example_bulk_operations()

        # Geochemical composition statistics
        print("\n--- Geochemical Composition Statistics ---")
        composition_stats = await storage.aggregate_geochemical_composition(
            sample_type="volcanic_rock"
        )

        if "composition_statistics" in composition_stats:
            print("Element statistics:")
            for stat in composition_stats["composition_statistics"][:5]:
                print(
                    f"  {stat['_id']}: avg={stat['avg_concentration']:.2f}%, "
                    f"count={stat['count']}"
                )

        # Temporal trends
        print("\n--- Temporal Trends ---")
        temporal_trends = await storage.aggregate_temporal_trends(
            collection_name="geochemical_samples",
            field_name="temperature",
            time_interval="day",
        )

        if "temporal_trends" in temporal_trends:
            print(
                f"Temperature trends over {len(temporal_trends['temporal_trends'])} days:"
            )
            for trend in temporal_trends["temporal_trends"][:3]:
                print(
                    f"  Day {trend['_id']}: avg={trend['avg_value']:.1f}K, "
                    f"count={trend['count']}"
                )

        # Spatial distribution
        print("\n--- Spatial Distribution ---")
        spatial_dist = await storage.aggregate_spatial_distribution(
            collection_name="geochemical_samples", grid_size=0.1
        )

        if "spatial_distribution" in spatial_dist:
            print(
                f"Data distributed across {len(spatial_dist['spatial_distribution'])} grid cells:"
            )
            for dist in spatial_dist["spatial_distribution"][:3]:
                print(
                    f"  Grid ({dist['_id']['lon']:.1f}, {dist['_id']['lat']:.1f}): "
                    f"{dist['count']} samples"
                )

    finally:
        await storage.cleanup()


async def example_data_quality_monitoring():
    """Example: Data quality monitoring and validation"""
    print("\n=== Data Quality Monitoring ===")

    storage = MongoDBStorage("mongodb://localhost:27017", "prometheus_examples")
    await storage.initialize()

    try:
        # Store samples with various quality issues
        problematic_samples = [
            {
                "timestamp": datetime.utcnow() + timedelta(hours=1),  # Future timestamp
                "location": {
                    "latitude": 200.0,
                    "longitude": -200.0,
                },  # Invalid coordinates
                "sample_type": "test",
                "temperature": -100.0,  # Invalid temperature
                "ph": 20.0,  # Invalid pH
                "source": "quality_test",
                "quality": "poor",
            },
            {
                "timestamp": datetime.utcnow(),
                "location": {"latitude": 45.0, "longitude": -121.0},
                "sample_type": "good_sample",
                "temperature": 1200.0,
                "ph": 7.0,
                "source": "quality_test",
                "quality": "excellent",
            },
        ]

        await storage.bulk_insert(problematic_samples, "geochemical_samples")

        # Generate quality report
        print("\n--- Data Quality Report ---")
        quality_report = await storage.get_data_quality_report("geochemical_samples")

        print(
            "Total documents:",
            quality_report.get("total_count", [{}])[0].get("count", 0),
        )

        print("\nQuality distribution:")
        for quality in quality_report.get("quality_distribution", []):
            print(f"  {quality['_id']}: {quality['count']} samples")

        print("\nValidation issues:")
        for issue in quality_report.get("validation_issues", []):
            print(f"  {issue['_id']}: {issue['count']} occurrences")

        print("\nData sources:")
        for source in quality_report.get("source_distribution", []):
            print(f"  {source['_id']}: {source['count']} samples")

    finally:
        await storage.cleanup()


async def example_custom_aggregation():
    """Example: Custom aggregation pipeline"""
    print("\n=== Custom Aggregation Pipeline ===")

    storage = MongoDBStorage("mongodb://localhost:27017", "prometheus_examples")
    await storage.initialize()

    try:
        # Ensure we have data
        await example_bulk_operations()

        # Custom aggregation: Find samples with high SiO2 content by location
        custom_pipeline = [
            {
                "$match": {
                    "composition.SiO2": {"$gte": 60.0},
                    "data_type": "geochemical_sample",
                }
            },
            {
                "$group": {
                    "_id": {
                        "lat_grid": {
                            "$round": [
                                {"$arrayElemAt": ["$location.coordinates", 1]},
                                1,
                            ]
                        },
                        "lon_grid": {
                            "$round": [
                                {"$arrayElemAt": ["$location.coordinates", 0]},
                                1,
                            ]
                        },
                    },
                    "count": {"$sum": 1},
                    "avg_sio2": {"$avg": "$composition.SiO2"},
                    "max_sio2": {"$max": "$composition.SiO2"},
                    "samples": {"$push": "$sample_id"},
                }
            },
            {"$sort": {"avg_sio2": -1}},
        ]

        collection = storage.collections.get(
            "geochemical_samples", "geochemical_samples"
        )
        cursor = storage.db[collection].aggregate(custom_pipeline)
        results = await cursor.to_list(length=None)

        print("High-SiO2 samples by location grid:")
        for result in results:
            print(
                f"  Grid ({result['_id']['lon_grid']}, {result['_id']['lat_grid']}): "
                f"{result['count']} samples, avg SiO2={result['avg_sio2']:.1f}%"
            )

    finally:
        await storage.cleanup()


async def main():
    """Run all examples"""
    print("MongoDB Storage Examples for Geological Data")
    print("=" * 50)

    try:
        await example_basic_storage()
        await example_unstructured_data()
        await example_bulk_operations()
        await example_advanced_queries()
        await example_aggregation_pipelines()
        await example_data_quality_monitoring()
        await example_custom_aggregation()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure MongoDB is running on localhost:27017")


if __name__ == "__main__":
    asyncio.run(main())
