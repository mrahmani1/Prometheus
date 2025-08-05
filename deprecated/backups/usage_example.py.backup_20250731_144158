#!/usr/bin/env python3
"""
Example usage of TimescaleDB service for geological data analysis
"""

import asyncio
import httpx
import json
from datetime import datetime, timedelta
import random
import numpy as np

# Service configuration
TIMESCALE_SERVICE_URL = "http://localhost:8080"

async def generate_sample_sensor_data(num_readings=1000):
    """Generate sample sensor data for testing"""
    readings = []
    base_time = datetime.utcnow() - timedelta(days=7)
    
    # Simulate different sensor types
    sensors = [
        {"id": "SEISMO_001", "type": "seismic_amplitude", "unit": "m/s", "base_value": 0.001},
        {"id": "TEMP_001", "type": "temperature", "unit": "celsius", "base_value": 25.0},
        {"id": "GPS_001", "type": "displacement", "unit": "mm", "base_value": 0.0},
        {"id": "GAS_001", "type": "so2_concentration", "unit": "ppm", "base_value": 10.0},
        {"id": "TILT_001", "type": "tilt_angle", "unit": "microradians", "base_value": 0.0}
    ]
    
    for i in range(num_readings):
        sensor = random.choice(sensors)
        timestamp = base_time + timedelta(minutes=i * 10)
        
        # Add some realistic variations and trends
        trend = 0.001 * i if sensor["type"] == "temperature" else 0
        noise = np.random.normal(0, sensor["base_value"] * 0.1)
        seasonal = sensor["base_value"] * 0.05 * np.sin(2 * np.pi * i / 144)  # Daily cycle
        
        value = sensor["base_value"] + trend + noise + seasonal
        
        # Add occasional anomalies
        if random.random() < 0.05:  # 5% chance of anomaly
            value += sensor["base_value"] * random.uniform(-2, 2)
        
        reading = {
            "sensor_id": sensor["id"],
            "timestamp": timestamp.isoformat() + "Z",
            "measurement_type": sensor["type"],
            "value": round(value, 6),
            "unit": sensor["unit"],
            "quality_flag": random.choice(["good"] * 9 + ["questionable"]),  # 90% good quality
            "location_lat": 37.7749 + random.uniform(-0.1, 0.1),
            "location_lon": -122.4194 + random.uniform(-0.1, 0.1)
        }
        
        readings.append(reading)
    
    return readings

async def insert_sample_data():
    """Insert sample data into TimescaleDB"""
    print("Generating sample sensor data...")
    readings = await generate_sample_sensor_data(1000)
    
    print(f"Inserting {len(readings)} sensor readings...")
    
    async with httpx.AsyncClient() as client:
        # Insert data in batches
        batch_size = 100
        for i in range(0, len(readings), batch_size):
            batch = readings[i:i + batch_size]
            
            response = await client.post(
                f"{TIMESCALE_SERVICE_URL}/sensor-readings/batch",
                json=batch,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Batch {i//batch_size + 1}: Inserted {result['inserted_count']} readings")
            else:
                print(f"Error inserting batch {i//batch_size + 1}: {response.status_code}")
                print(response.text)

async def query_time_series_data():
    """Query time-series data with different aggregations"""
    print("\nQuerying time-series data...")
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)
    
    queries = [
        {
            "name": "Raw temperature data",
            "query": {
                "sensor_ids": ["TEMP_001"],
                "start_time": start_time.isoformat() + "Z",
                "end_time": end_time.isoformat() + "Z",
                "measurement_types": ["temperature"]
            }
        },
        {
            "name": "Hourly average temperature",
            "query": {
                "sensor_ids": ["TEMP_001"],
                "start_time": start_time.isoformat() + "Z",
                "end_time": end_time.isoformat() + "Z",
                "measurement_types": ["temperature"],
                "aggregation": "avg",
                "interval": "1h"
            }
        },
        {
            "name": "All seismic sensors - max values",
            "query": {
                "sensor_ids": ["SEISMO_001"],
                "start_time": start_time.isoformat() + "Z",
                "end_time": end_time.isoformat() + "Z",
                "measurement_types": ["seismic_amplitude"],
                "aggregation": "max",
                "interval": "1h"
            }
        }
    ]
    
    async with httpx.AsyncClient() as client:
        for query_info in queries:
            print(f"\n{query_info['name']}:")
            
            response = await client.post(
                f"{TIMESCALE_SERVICE_URL}/query/time-series",
                json=query_info["query"],
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  Status: {result['status']}")
                print(f"  Data points: {result['count']}")
                
                if result['data']:
                    sample = result['data'][0]
                    print(f"  Sample data: {json.dumps(sample, indent=4, default=str)}")
            else:
                print(f"  Error: {response.status_code}")
                print(f"  {response.text}")

async def analyze_geological_trends():
    """Perform geological trend analysis"""
    print("\nPerforming geological trend analysis...")
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)
    
    analyses = [
        {
            "sensor_id": "TEMP_001",
            "measurement_type": "temperature",
            "description": "Temperature trend analysis"
        },
        {
            "sensor_id": "SEISMO_001",
            "measurement_type": "seismic_amplitude",
            "description": "Seismic activity trend analysis"
        },
        {
            "sensor_id": "GPS_001",
            "measurement_type": "displacement",
            "description": "Ground displacement trend analysis"
        }
    ]
    
    async with httpx.AsyncClient() as client:
        for analysis_config in analyses:
            print(f"\n{analysis_config['description']}:")
            
            request_data = {
                "sensor_id": analysis_config["sensor_id"],
                "measurement_type": analysis_config["measurement_type"],
                "start_time": start_time.isoformat() + "Z",
                "end_time": end_time.isoformat() + "Z",
                "trend_window": "1d"
            }
            
            response = await client.post(
                f"{TIMESCALE_SERVICE_URL}/analysis/trend",
                json=request_data,
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display trend analysis
                trend = result["trend_analysis"]
                print(f"  Trend slope: {trend['slope']:.6f}")
                print(f"  R-squared: {trend['r_squared']:.3f}")
                print(f"  Data points: {trend['data_points']}")
                print(f"  Mean value: {trend['mean_value']:.3f}")
                
                # Display anomaly detection
                anomaly = result["anomaly_detection"]
                print(f"  Anomalies: {anomaly['anomaly_count']}/{anomaly['total_points']} ({anomaly['anomaly_rate']:.1%})")
                print(f"  Max Z-score: {anomaly['max_z_score']:.2f}")
                
                # Display geological interpretation
                interpretation = result["geological_interpretation"]
                print(f"  Trend significance: {interpretation['trend_significance']}")
                print(f"  Anomaly level: {interpretation['anomaly_level']}")
                
                if interpretation["geological_implications"]:
                    print("  Geological implications:")
                    for implication in interpretation["geological_implications"]:
                        print(f"    - {implication}")
                
                if interpretation["recommendations"]:
                    print("  Recommendations:")
                    for recommendation in interpretation["recommendations"]:
                        print(f"    - {recommendation}")
            else:
                print(f"  Error: {response.status_code}")
                print(f"  {response.text}")

async def get_database_statistics():
    """Get database performance statistics"""
    print("\nRetrieving database statistics...")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TIMESCALE_SERVICE_URL}/statistics/database",
            timeout=30.0
        )
        
        if response.status_code == 200:
            stats = response.json()
            
            print("Hypertable Statistics:")
            for table_stat in stats["hypertable_statistics"]:
                print(f"  {table_stat['hypertable_name']}:")
                print(f"    Chunks: {table_stat['num_chunks']}")
                print(f"    Total size: {table_stat['total_bytes'] / 1024 / 1024:.2f} MB")
            
            print("\nCompression Statistics:")
            for comp_stat in stats["compression_statistics"]:
                if comp_stat.get("compressed_total_bytes"):
                    compression_ratio = comp_stat["uncompressed_total_bytes"] / comp_stat["compressed_total_bytes"]
                    print(f"  {comp_stat['hypertable_name']}: {compression_ratio:.1f}x compression")
            
            print("\nRecent Ingestion (last hour):")
            for ing_stat in stats["ingestion_statistics"]:
                print(f"  {ing_stat['table_name']}: {ing_stat['recent_inserts']} new records")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

async def test_manual_compression():
    """Test manual compression functionality"""
    print("\nTesting manual compression...")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TIMESCALE_SERVICE_URL}/maintenance/compress",
            timeout=60.0
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Compression status: {result['status']}")
            print(f"Compressed chunks: {result['count']}")
            
            if result['compressed_chunks']:
                print("Compressed chunks:")
                for chunk in result['compressed_chunks']:
                    print(f"  - {chunk}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

async def check_service_health():
    """Check service health and readiness"""
    print("Checking service health...")
    
    async with httpx.AsyncClient() as client:
        # Health check
        health_response = await client.get(f"{TIMESCALE_SERVICE_URL}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"Health: {health_data['status']}")
        else:
            print(f"Health check failed: {health_response.status_code}")
            return False
        
        # Readiness check
        ready_response = await client.get(f"{TIMESCALE_SERVICE_URL}/ready")
        if ready_response.status_code == 200:
            ready_data = ready_response.json()
            print(f"Readiness: {ready_data['status']}")
        else:
            print(f"Readiness check failed: {ready_response.status_code}")
            return False
    
    return True

async def main():
    """Main example function"""
    print("TimescaleDB Service Usage Example")
    print("=" * 40)
    
    # Check if service is healthy
    if not await check_service_health():
        print("Service is not healthy. Please check the service status.")
        return
    
    try:
        # Insert sample data
        await insert_sample_data()
        
        # Query data
        await query_time_series_data()
        
        # Analyze trends
        await analyze_geological_trends()
        
        # Get statistics
        await get_database_statistics()
        
        # Test compression
        await test_manual_compression()
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Error during example execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())