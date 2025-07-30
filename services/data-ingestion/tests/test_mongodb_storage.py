"""Tests for MongoDB storage functionality"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import motor.motor_asyncio
from pymongo.errors import ServerSelectionTimeoutError

from storage.mongodb_storage import MongoDBStorage
from models.geological_data import (
    GeochemicalSample, AtmosphericMeasurement, GeologicalLocation, DataQuality
)
from models.unstructured_geological_data import RockSample, RockType

# Test configuration
TEST_MONGODB_URL = "mongodb://localhost:27017"
TEST_DATABASE = "prometheus_test"

@pytest.fixture
async def mongodb_storage():
    """Create MongoDB storage instance for testing"""
    storage = MongoDBStorage(TEST_MONGODB_URL, TEST_DATABASE)
    
    try:
        await storage.initialize()
        yield storage
    except ServerSelectionTimeoutError:
        pytest.skip("MongoDB not available for testing")
    finally:
        # Cleanup test database
        if storage.client:
            await storage.client.drop_database(TEST_DATABASE)
            await storage.cleanup()

@pytest.fixture
def sample_location():
    """Sample geological location for testing"""
    return GeologicalLocation(
        latitude=45.0,
        longitude=-120.0,
        elevation=1000.0,
        depth=10.0,
        coordinate_system="WGS84",
        uncertainty_m=5.0
    )

@pytest.fixture
def sample_geochemical_sample(sample_location):
    """Sample geochemical sample for testing"""
    return GeochemicalSample(
        timestamp=datetime.utcnow(),
        location=sample_location,
        sample_type="rock",
        composition={
            "SiO2": 65.5,
            "Al2O3": 15.2,
            "FeO": 8.1,
            "MgO": 3.2,
            "CaO": 4.8,
            "Na2O": 2.1,
            "K2O": 1.1
        },
        isotope_ratios={
            "Sr87_Sr86": 0.7045,
            "Nd143_Nd144": 0.5126
        },
        temperature=1200.0,
        pressure=1.5e9,
        ph=7.2,
        source="test_lab",
        quality=DataQuality.GOOD
    )

@pytest.fixture
def sample_atmospheric_measurement(sample_location):
    """Sample atmospheric measurement for testing"""
    return AtmosphericMeasurement(
        timestamp=datetime.utcnow(),
        location=sample_location,
        pressure=101325.0,
        temperature=298.15,
        humidity=65.0,
        wind_speed=5.2,
        wind_direction=180.0,
        gas_concentrations={
            "CO2": 410.0,
            "SO2": 0.05,
            "H2S": 0.01
        },
        source="weather_station",
        quality=DataQuality.EXCELLENT
    )

class TestMongoDBStorage:
    """Test MongoDB storage functionality"""
    
    async def test_initialization(self, mongodb_storage):
        """Test MongoDB storage initialization"""
        assert mongodb_storage.is_initialized
        assert mongodb_storage.client is not None
        assert mongodb_storage.db is not None
        
        # Test health check
        is_healthy = await mongodb_storage.is_healthy()
        assert is_healthy
    
    async def test_store_geochemical_sample(self, mongodb_storage, sample_geochemical_sample):
        """Test storing geochemical sample"""
        document_id = await mongodb_storage.store_geochemical_sample(sample_geochemical_sample)
        
        assert document_id is not None
        assert len(document_id) > 0
        
        # Verify document was stored
        collection = mongodb_storage.db[mongodb_storage.collections['geochemical_samples']]
        doc = await collection.find_one({"_id": mongodb_storage.client.db.ObjectId(document_id)})
        
        assert doc is not None
        assert doc["sample_type"] == "rock"
        assert doc["composition"]["SiO2"] == 65.5
        assert doc["location"]["type"] == "Point"
        assert len(doc["location"]["coordinates"]) == 3
    
    async def test_store_atmospheric_measurement(self, mongodb_storage, sample_atmospheric_measurement):
        """Test storing atmospheric measurement"""
        document_id = await mongodb_storage.store_atmospheric_measurement(sample_atmospheric_measurement)
        
        assert document_id is not None
        assert len(document_id) > 0
        
        # Verify document was stored
        collection = mongodb_storage.db[mongodb_storage.collections['atmospheric_measurements']]
        doc = await collection.find_one({"_id": mongodb_storage.client.db.ObjectId(document_id)})
        
        assert doc is not None
        assert doc["pressure"] == 101325.0
        assert doc["gas_concentrations"]["CO2"] == 410.0
    
    async def test_store_unstructured_data(self, mongodb_storage):
        """Test storing arbitrary unstructured data"""
        test_data = {
            "timestamp": datetime.utcnow(),
            "location": {
                "latitude": 40.0,
                "longitude": -110.0,
                "elevation": 2000.0
            },
            "rock_type": "granite",
            "mineral_content": ["quartz", "feldspar", "mica"],
            "custom_field": "test_value",
            "nested_data": {
                "analysis_method": "XRF",
                "analyst": "Dr. Smith"
            }
        }
        
        document_id = await mongodb_storage.store_unstructured_data(test_data, "rock_samples")
        
        assert document_id is not None
        
        # Verify document was stored with GeoJSON location
        collection = mongodb_storage.db["rock_samples"]
        doc = await collection.find_one({"_id": mongodb_storage.client.db.ObjectId(document_id)})
        
        assert doc is not None
        assert doc["rock_type"] == "granite"
        assert doc["location"]["type"] == "Point"
        assert doc["location"]["coordinates"] == [-110.0, 40.0, 2000.0]
    
    async def test_bulk_insert(self, mongodb_storage):
        """Test bulk insert functionality"""
        test_documents = [
            {
                "timestamp": datetime.utcnow(),
                "sample_id": f"BULK_{i}",
                "value": i * 10.0,
                "category": "test"
            }
            for i in range(5)
        ]
        
        document_ids = await mongodb_storage.bulk_insert(test_documents, "bulk_test")
        
        assert len(document_ids) == 5
        
        # Verify documents were stored
        collection = mongodb_storage.db["bulk_test"]
        count = await collection.count_documents({"category": "test"})
        assert count == 5
    
    async def test_query_by_location(self, mongodb_storage, sample_geochemical_sample):
        """Test location-based queries"""
        # Store a sample first
        await mongodb_storage.store_geochemical_sample(sample_geochemical_sample)
        
        # Query by location
        results = await mongodb_storage.query_by_location(
            longitude=-120.0,
            latitude=45.0,
            radius_km=10.0,
            collection_name="geochemical_samples",
            limit=10
        )
        
        assert len(results) >= 1
        assert results[0]["sample_type"] == "rock"
    
    async def test_query_by_time_range(self, mongodb_storage, sample_geochemical_sample):
        """Test time range queries"""
        # Store a sample first
        await mongodb_storage.store_geochemical_sample(sample_geochemical_sample)
        
        # Query by time range
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow() + timedelta(hours=1)
        
        results = await mongodb_storage.query_by_time_range(
            start_time=start_time,
            end_time=end_time,
            collection_name="geochemical_samples",
            limit=10
        )
        
        assert len(results) >= 1
        assert results[0]["sample_type"] == "rock"
    
    async def test_text_search(self, mongodb_storage):
        """Test full-text search functionality"""
        # Store some test data with searchable text
        test_data = {
            "timestamp": datetime.utcnow(),
            "description": "granite sample with quartz and feldspar minerals",
            "location_name": "Mount Rushmore",
            "notes": "collected during field survey"
        }
        
        await mongodb_storage.store_unstructured_data(test_data, "field_observations")
        
        # Wait a moment for indexing
        await asyncio.sleep(0.1)
        
        # Search for text
        results = await mongodb_storage.text_search("granite quartz", "field_observations")
        
        assert len(results) >= 1
        assert "granite" in results[0]["description"]
    
    async def test_aggregate_geochemical_composition(self, mongodb_storage, sample_geochemical_sample):
        """Test geochemical composition aggregation"""
        # Store multiple samples
        for i in range(3):
            sample = sample_geochemical_sample
            sample.composition["SiO2"] = 60.0 + i * 2.0
            await mongodb_storage.store_geochemical_sample(sample)
        
        # Aggregate composition data
        results = await mongodb_storage.aggregate_geochemical_composition(
            sample_type="rock"
        )
        
        assert "composition_statistics" in results
        assert len(results["composition_statistics"]) > 0
        
        # Check that SiO2 statistics are present
        sio2_stats = next((stat for stat in results["composition_statistics"] 
                          if stat["_id"] == "SiO2"), None)
        assert sio2_stats is not None
        assert sio2_stats["count"] >= 3
    
    async def test_aggregate_temporal_trends(self, mongodb_storage, sample_atmospheric_measurement):
        """Test temporal trends aggregation"""
        # Store measurements with different timestamps
        for i in range(3):
            measurement = sample_atmospheric_measurement
            measurement.timestamp = datetime.utcnow() - timedelta(days=i)
            measurement.temperature = 298.15 + i * 2.0
            await mongodb_storage.store_atmospheric_measurement(measurement)
        
        # Aggregate temporal trends
        results = await mongodb_storage.aggregate_temporal_trends(
            collection_name="atmospheric_measurements",
            field_name="temperature",
            time_interval="day"
        )
        
        assert "temporal_trends" in results
        assert len(results["temporal_trends"]) > 0
    
    async def test_aggregate_spatial_distribution(self, mongodb_storage, sample_geochemical_sample):
        """Test spatial distribution aggregation"""
        # Store samples at different locations
        for i in range(3):
            sample = sample_geochemical_sample
            sample.location.latitude = 45.0 + i * 0.1
            sample.location.longitude = -120.0 + i * 0.1
            await mongodb_storage.store_geochemical_sample(sample)
        
        # Aggregate spatial distribution
        results = await mongodb_storage.aggregate_spatial_distribution(
            collection_name="geochemical_samples",
            grid_size=0.1
        )
        
        assert "spatial_distribution" in results
        assert len(results["spatial_distribution"]) > 0
    
    async def test_data_quality_report(self, mongodb_storage, sample_geochemical_sample):
        """Test data quality report generation"""
        # Store samples with different quality levels
        qualities = [DataQuality.EXCELLENT, DataQuality.GOOD, DataQuality.FAIR]
        for quality in qualities:
            sample = sample_geochemical_sample
            sample.quality = quality
            await mongodb_storage.store_geochemical_sample(sample)
        
        # Generate quality report
        report = await mongodb_storage.get_data_quality_report("geochemical_samples")
        
        assert "total_count" in report
        assert "quality_distribution" in report
        assert "temporal_coverage" in report
        assert "source_distribution" in report
        
        # Check quality distribution
        quality_dist = report["quality_distribution"]
        assert len(quality_dist) > 0
    
    async def test_data_validation(self, mongodb_storage, sample_location):
        """Test data validation functionality"""
        # Test with invalid data
        invalid_sample = GeochemicalSample(
            timestamp=datetime.utcnow() + timedelta(days=1),  # Future timestamp
            location=GeologicalLocation(
                latitude=200.0,  # Invalid latitude
                longitude=-200.0,  # Invalid longitude
                elevation=1000.0
            ),
            sample_type="rock",
            composition={"SiO2": 65.5},
            temperature=-100.0,  # Invalid temperature
            ph=20.0,  # Invalid pH
            source="test",
            quality=DataQuality.POOR
        )
        
        document_id = await mongodb_storage.store_geochemical_sample(invalid_sample)
        
        # Verify document was stored with validation issues
        collection = mongodb_storage.db[mongodb_storage.collections['geochemical_samples']]
        doc = await collection.find_one({"_id": mongodb_storage.client.db.ObjectId(document_id)})
        
        assert doc is not None
        assert "validation_issues" in doc
        assert len(doc["validation_issues"]) > 0
    
    async def test_cleanup_old_data(self, mongodb_storage, sample_geochemical_sample):
        """Test old data cleanup functionality"""
        # Store old sample
        old_sample = sample_geochemical_sample
        old_sample.timestamp = datetime.utcnow() - timedelta(days=400)
        await mongodb_storage.store_geochemical_sample(old_sample)
        
        # Store recent sample
        recent_sample = sample_geochemical_sample
        recent_sample.timestamp = datetime.utcnow()
        await mongodb_storage.store_geochemical_sample(recent_sample)
        
        # Cleanup old data (keep only 365 days)
        deleted_count = await mongodb_storage.cleanup_old_data("geochemical_samples", 365)
        
        assert deleted_count >= 1
        
        # Verify only recent data remains
        collection = mongodb_storage.db[mongodb_storage.collections['geochemical_samples']]
        remaining_count = await collection.count_documents({})
        assert remaining_count >= 1  # Recent sample should remain

@pytest.mark.asyncio
async def test_mongodb_storage_without_server():
    """Test MongoDB storage behavior when server is not available"""
    storage = MongoDBStorage("mongodb://nonexistent:27017", "test_db")
    
    with pytest.raises(Exception):
        await storage.initialize()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])