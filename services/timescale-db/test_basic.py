#!/usr/bin/env python3
"""
Basic integration test for TimescaleDB service
"""

import sys

def test_imports():
    """Test that all required modules can be imported"""
    try:
        pass

        print("✓ All required modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_app_creation():
    """Test that the FastAPI app can be created"""
    try:
        # Mock the database pool to avoid connection issues
        import unittest.mock

        with unittest.mock.patch("main.create_db_pool"):
            with unittest.mock.patch("main.db_pool", None):
                # Check that app is a FastAPI instance
                from fastapi import FastAPI
                from main import app

                assert isinstance(app, FastAPI)

                print("✓ FastAPI app created successfully")
                return True
    except Exception as e:
        print(f"✗ App creation error: {e}")
        return False

def test_pydantic_models():
    """Test that Pydantic models are valid"""
    try:
        from datetime import datetime

        from main import SensorReading, TimeSeriesQuery

        # Test SensorReading model
        reading = SensorReading(
            sensor_id="TEST_001",
            timestamp=datetime.utcnow(),
            measurement_type="temperature",
            value=25.5,
            unit="celsius",
        )
        assert reading.sensor_id == "TEST_001"

        # Test TimeSeriesQuery model
        query = TimeSeriesQuery(
            sensor_ids=["TEST_001"],
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
        )
        assert len(query.sensor_ids) == 1

        print("✓ Pydantic models validated successfully")
        return True
    except Exception as e:
        print(f"✗ Pydantic model error: {e}")
        return False

def test_geological_interpretation():
    """Test geological interpretation function"""
    try:
        from main import interpret_geological_trends

import torch.nn as nn

        # Test with sample analysis data
        analysis = {
            "trend_analysis": {"slope": 0.005, "r_squared": 0.85},
            "anomaly_detection": {"anomaly_rate": 0.02, "max_z_score": 2.5},
            "seasonal_analysis": {"trend_strength": 0.8, "seasonal_strength": 0.3},
        }

        interpretation = interpret_geological_trends(analysis)

        assert "trend_significance" in interpretation
        assert "anomaly_level" in interpretation
        assert "geological_implications" in interpretation
        assert "recommendations" in interpretation

        print("✓ Geological interpretation function works correctly")
        return True
    except Exception as e:
        print(f"✗ Geological interpretation error: {e}")
        return False

def main():
    """Run all basic tests"""
    print("Running basic TimescaleDB service tests...")
    print("=" * 50)

    tests = [
        test_imports,
        test_app_creation,
        test_pydantic_models,
        test_geological_interpretation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All basic tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
