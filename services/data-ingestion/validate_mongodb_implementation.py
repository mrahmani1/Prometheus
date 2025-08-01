"""Validation script for MongoDB implementation"""

import sys
import traceback
from datetime import datetime

# Test imports
try:
    except Exception:
        pass
        AtmosphericMeasurement,
        DataQuality,
        GeochemicalSample,
        GeologicalLocation,
    )
        CrystalSystem,
        MineralAnalysis,
        RockSample,
        RockType,
    )

import torch.nn as nn
import torch.nn.functional as F

    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_data_models():
    """Test data model creation"""
    try:
        # Test GeologicalLocation
        location = GeologicalLocation(
            latitude=45.0, longitude=-120.0, elevation=1000.0, depth=10.0
        )
        print("✓ GeologicalLocation model works")

        # Test GeochemicalSample
        sample = GeochemicalSample(
            timestamp=datetime.utcnow(),
            location=location,
            sample_type="rock",
            composition={"SiO2": 65.5, "Al2O3": 15.2},
            temperature=1200.0,
            pressure=1.5e9,
            ph=7.2,
            source="test",
            quality=DataQuality.GOOD,
        )
        print("✓ GeochemicalSample model works")

        # Test AtmosphericMeasurement
        measurement = AtmosphericMeasurement(
            timestamp=datetime.utcnow(),
            location=location,
            pressure=101325.0,
            temperature=298.15,
            humidity=65.0,
            source="test",
            quality=DataQuality.EXCELLENT,
        )
        print("✓ AtmosphericMeasurement model works")

        # Test RockSample
        rock_sample = RockSample(
            timestamp=datetime.utcnow(),
            location=location,
            sample_id="TEST_001",
            rock_type=RockType.IGNEOUS,
            formation="Test Formation",
            age_ma=100.0,
            source="test",
            quality=DataQuality.GOOD,
        )
        print("✓ RockSample model works")

        # Test MineralAnalysis
        mineral = MineralAnalysis(
            timestamp=datetime.utcnow(),
            location=location,
            sample_id="MIN_001",
            mineral_name="quartz",
            chemical_formula="SiO2",
            crystal_system=CrystalSystem.HEXAGONAL,
            hardness=7.0,
            source="test",
            quality=DataQuality.EXCELLENT,
        )
        print("✓ MineralAnalysis model works")

        return True
    except Exception as e:
        print(f"✗ Data model error: {e}")
        traceback.print_exc()
        return False

def test_mongodb_storage_class():
    """Test MongoDB storage class instantiation"""
    try:
        storage = MongoDBStorage("mongodb://localhost:27017", "test_db")
        print("✓ MongoDBStorage class instantiation works")

        # Test method existence
        methods_to_check = [
            "initialize",
            "store_geochemical_sample",
            "store_atmospheric_measurement",
            "store_unstructured_data",
            "bulk_insert",
            "query_by_location",
            "query_by_time_range",
            "text_search",
            "aggregate_geochemical_composition",
            "aggregate_temporal_trends",
            "aggregate_spatial_distribution",
            "get_data_quality_report",
            "cleanup_old_data",
            "is_healthy",
            "cleanup",
        ]

        for method in methods_to_check:
            if not hasattr(storage, method):
                print(f"✗ Missing method: {method}")
                return False
            print(f"✓ Method {method} exists")

        return True
    except Exception as e:
        print(f"✗ MongoDB storage class error: {e}")
        traceback.print_exc()
        return False

def test_validation_functions():
    """Test data validation functions"""
    try:
        storage = MongoDBStorage("mongodb://localhost:27017", "test_db")

        # Test location conversion
        location = GeologicalLocation(latitude=45.0, longitude=-120.0, elevation=1000.0)
        geojson_location = storage._convert_to_geojson_location(location)

        expected_geojson = {"type": "Point", "coordinates": [-120.0, 45.0, 1000.0]}

        if geojson_location == expected_geojson:
            print("✓ GeoJSON location conversion works")
        else:
            print(f"✗ GeoJSON conversion failed: {geojson_location}")
            return False

        # Test data validation
        test_data = {
            "timestamp": datetime.utcnow(),
            "location": geojson_location,
            "data_type": "test",
            "temperature": 1200.0,
            "ph": 7.0,
        }

        issues = storage._validate_data_quality(test_data)
        print(f"✓ Data validation works (found {len(issues)} issues)")

        # Test with invalid data
        invalid_data = {
            "timestamp": datetime.utcnow(),
            "location": {
                "type": "Point",
                "coordinates": [200.0, 100.0],  # Invalid coordinates
            },
            "data_type": "test",
            "temperature": -100.0,  # Invalid temperature
            "ph": 20.0,  # Invalid pH
        }

        issues = storage._validate_data_quality(invalid_data)
        if len(issues) > 0:
            print(
                f"✓ Data validation correctly identifies issues: {len(issues)} issues found"
            )
        else:
            print("✗ Data validation failed to identify invalid data")
            return False

        return True
    except Exception as e:
        print(f"✗ Validation function error: {e}")
        traceback.print_exc()
        return False

def test_api_router():
    """Test API router configuration"""
    try:
        # Check if router has expected routes
        routes = [route.path for route in router.routes]

        expected_routes = [
            "/geochemical-samples",
            "/atmospheric-measurements",
            "/unstructured-data",
            "/bulk-insert",
            "/query/location",
            "/query/time-range",
            "/search",
            "/aggregate/geochemical-composition",
            "/aggregate/temporal-trends",
            "/aggregate/spatial-distribution",
            "/quality-report/{collection_name}",
            "/cleanup/{collection_name}",
            "/collections",
            "/health",
        ]

        for expected_route in expected_routes:
            # Check if any route contains the expected path
            if any(expected_route in route for route in routes):
                print(f"✓ API route {expected_route} exists")
            else:
                print(f"✗ Missing API route: {expected_route}")
                return False

        return True
    except Exception as e:
        print(f"✗ API router error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    print("MongoDB Implementation Validation")
    print("=" * 40)

    tests = [
        ("Data Models", test_data_models),
        ("MongoDB Storage Class", test_mongodb_storage_class),
        ("Validation Functions", test_validation_functions),
        ("API Router", test_api_router),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        if test_func():
            passed += 1
            print(f"✓ {test_name} passed")
        else:
            print(f"✗ {test_name} failed")

    print("\n" + "=" * 40)
    print(f"Validation Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All validation tests passed!")
        print("\nMongoDB implementation is ready for use.")
        print("\nNext steps:")
        print("1. Start MongoDB server: docker run -d -p 27017:27017 mongo:latest")
        print("2. Run the examples: python examples/mongodb_usage_examples.py")
        print("3. Run the full test suite with MongoDB running")
        return True
    else:
        print("❌ Some validation tests failed.")
        print("Please fix the issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
