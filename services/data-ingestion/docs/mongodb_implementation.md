# MongoDB Implementation for Unstructured Geological Data

## Overview

This document describes the MongoDB implementation for storing and managing unstructured geological data in the Prometheus AI Geodynamic Platform. The implementation provides flexible schemas, advanced indexing strategies, aggregation pipelines, and comprehensive data validation for diverse geological sample data.

## Architecture

### Core Components

1. **MongoDBStorage Class** (`storage/mongodb_storage.py`)
   - Main storage interface for MongoDB operations
   - Handles connection management and initialization
   - Provides data validation and quality control
   - Implements advanced querying and aggregation

2. **Extended Data Models** (`models/unstructured_geological_data.py`)
   - Comprehensive data models for various geological data types
   - Support for rock samples, mineral analyses, fluid samples, core samples
   - Field observations, laboratory results, and geophysical surveys

3. **API Endpoints** (`api/mongodb_api.py`)
   - RESTful API for MongoDB operations
   - Bulk operations support
   - Advanced querying capabilities
   - Data quality reporting

## Features

### Flexible Schema Design

The MongoDB implementation supports flexible schemas for diverse geological data types:

- **Geochemical Samples**: Chemical composition, isotope ratios, temperature, pressure
- **Rock Samples**: Lithology, texture, mineral assemblage, structural features
- **Mineral Analyses**: Crystal structure, optical properties, chemical composition
- **Fluid Samples**: Water, gas, oil analysis with chemical and isotopic data
- **Core Samples**: Drill core data with depth intervals and geological descriptions
- **Field Observations**: Outcrop descriptions, structural measurements, photos
- **Laboratory Results**: Analytical results with quality control metrics

### Advanced Indexing Strategies

#### Spatial Indexes
- **GeoJSON Point indexing** for location-based queries
- **2D geospatial indexes** for efficient spatial operations
- **Compound spatial-temporal indexes** for spatio-temporal queries

#### Temporal Indexes
- **Descending timestamp indexes** for recent data retrieval
- **Compound time-location indexes** for efficient filtering

#### Content-Specific Indexes
- **Chemical composition indexes** (SiO2, Al2O3, FeO, etc.)
- **Isotope ratio indexes** (Sr87/Sr86, Nd143/Nd144, etc.)
- **Physical property indexes** (temperature, pressure, pH)
- **Full-text search indexes** for flexible content queries

#### Example Index Creation
```javascript
// Geochemical composition indexes
db.geochemical_samples.createIndex({"composition.SiO2": 1})
db.geochemical_samples.createIndex({"composition.Al2O3": 1})
db.geochemical_samples.createIndex({"isotope_ratios.Sr87_Sr86": 1})

// Spatial-temporal compound index
db.geochemical_samples.createIndex({
    "timestamp": -1,
    "location.coordinates": "2d"
})

// Full-text search
db.geochemical_samples.createIndex({"$**": "text"})
```

### Data Validation and Quality Control

#### Schema Validation
- **JSON Schema validation** at the database level
- **Field type validation** (dates, coordinates, numerical ranges)
- **Required field enforcement**
- **Custom validation rules** for geological data

#### Quality Assessment
- **Automatic data quality scoring** based on completeness and validity
- **Validation issue tracking** with detailed error messages
- **Quality metrics calculation** for monitoring data health
- **Outlier detection** for numerical fields

#### Example Validation Schema
```json
{
  "$jsonSchema": {
    "bsonType": "object",
    "required": ["timestamp", "location", "sample_type", "data_type"],
    "properties": {
      "timestamp": {"bsonType": "date"},
      "location": {
        "bsonType": "object",
        "required": ["type", "coordinates"],
        "properties": {
          "type": {"enum": ["Point"]},
          "coordinates": {
            "bsonType": "array",
            "minItems": 2,
            "maxItems": 3,
            "items": {"bsonType": "double"}
          }
        }
      },
      "temperature": {
        "bsonType": ["double", "null"],
        "minimum": 0,
        "maximum": 2000
      },
      "ph": {
        "bsonType": ["double", "null"],
        "minimum": 0,
        "maximum": 14
      }
    }
  }
}
```

### Aggregation Pipelines

#### Geochemical Composition Analysis
```python
# Statistical analysis of chemical composition
pipeline = [
    {"$match": {"data_type": "geochemical_sample"}},
    {"$unwind": {"path": "$composition"}},
    {
        "$group": {
            "_id": "$composition.k",
            "count": {"$sum": 1},
            "avg_concentration": {"$avg": "$composition.v"},
            "min_concentration": {"$min": "$composition.v"},
            "max_concentration": {"$max": "$composition.v"},
            "std_dev": {"$stdDevPop": "$composition.v"}
        }
    },
    {"$sort": {"avg_concentration": -1}}
]
```

#### Temporal Trend Analysis
```python
# Analyze temporal trends for specific fields
pipeline = [
    {"$match": {field_name: {"$exists": True, "$ne": None}}},
    {
        "$group": {
            "_id": {
                "year": {"$year": "$timestamp"},
                "month": {"$month": "$timestamp"},
                "day": {"$dayOfMonth": "$timestamp"}
            },
            "count": {"$sum": 1},
            "avg_value": {"$avg": f"${field_name}"},
            "min_value": {"$min": f"${field_name}"},
            "max_value": {"$max": f"${field_name}"}
        }
    },
    {"$sort": {"_id": 1}}
]
```

#### Spatial Distribution Analysis
```python
# Analyze spatial distribution on a grid
pipeline = [
    {"$match": {"location": {"$exists": True}}},
    {
        "$addFields": {
            "grid_lon": {
                "$multiply": [
                    {"$floor": {"$divide": [{"$arrayElemAt": ["$location.coordinates", 0]}, grid_size]}},
                    grid_size
                ]
            },
            "grid_lat": {
                "$multiply": [
                    {"$floor": {"$divide": [{"$arrayElemAt": ["$location.coordinates", 1]}, grid_size]}},
                    grid_size
                ]
            }
        }
    },
    {
        "$group": {
            "_id": {"lon": "$grid_lon", "lat": "$grid_lat"},
            "count": {"$sum": 1},
            "data_types": {"$addToSet": "$data_type"}
        }
    }
]
```

## API Endpoints

### Data Storage
- `POST /api/v1/mongodb/geochemical-samples` - Store geochemical sample
- `POST /api/v1/mongodb/atmospheric-measurements` - Store atmospheric measurement
- `POST /api/v1/mongodb/unstructured-data` - Store arbitrary geological data
- `POST /api/v1/mongodb/bulk-insert` - Bulk insert multiple documents

### Querying
- `POST /api/v1/mongodb/query/location` - Query by geographic location
- `POST /api/v1/mongodb/query/time-range` - Query by time range
- `GET /api/v1/mongodb/search` - Full-text search

### Aggregation
- `GET /api/v1/mongodb/aggregate/geochemical-composition` - Composition statistics
- `GET /api/v1/mongodb/aggregate/temporal-trends` - Temporal trend analysis
- `GET /api/v1/mongodb/aggregate/spatial-distribution` - Spatial distribution
- `POST /api/v1/mongodb/aggregate/custom` - Custom aggregation pipeline

### Management
- `GET /api/v1/mongodb/quality-report/{collection}` - Data quality report
- `DELETE /api/v1/mongodb/cleanup/{collection}` - Clean up old data
- `GET /api/v1/mongodb/collections` - List available collections
- `GET /api/v1/mongodb/health` - Health check

## Usage Examples

### Storing Geochemical Data
```python
from models.geological_data import GeochemicalSample, GeologicalLocation, DataQuality

sample = GeochemicalSample(
    timestamp=datetime.utcnow(),
    location=GeologicalLocation(
        latitude=45.3311,
        longitude=-121.7113,
        elevation=3429.0
    ),
    sample_type="volcanic_rock",
    composition={
        "SiO2": 58.2,
        "Al2O3": 17.1,
        "FeO": 8.9,
        "MgO": 4.2
    },
    isotope_ratios={
        "Sr87_Sr86": 0.7038,
        "Nd143_Nd144": 0.5129
    },
    temperature=1150.0,
    pressure=2.1e8,
    source="USGS_Survey",
    quality=DataQuality.EXCELLENT
)

document_id = await storage.store_geochemical_sample(sample)
```

### Location-Based Query
```python
# Find samples within 50km of Mount Hood
results = await storage.query_by_location(
    longitude=-121.7113,
    latitude=45.3311,
    radius_km=50.0,
    collection_name="geochemical_samples",
    limit=100
)
```

### Bulk Insert
```python
# Insert multiple samples efficiently
samples = [
    {
        "timestamp": datetime.utcnow(),
        "location": {"latitude": 45.0, "longitude": -121.0},
        "sample_type": "rock",
        "composition": {"SiO2": 65.0, "Al2O3": 15.0}
    }
    # ... more samples
]

document_ids = await storage.bulk_insert(samples, "geochemical_samples")
```

### Custom Aggregation
```python
# Find high-silica samples by location
pipeline = [
    {"$match": {"composition.SiO2": {"$gte": 70.0}}},
    {
        "$group": {
            "_id": {
                "lat": {"$round": [{"$arrayElemAt": ["$location.coordinates", 1]}, 1]},
                "lon": {"$round": [{"$arrayElemAt": ["$location.coordinates", 0]}, 1]}
            },
            "count": {"$sum": 1},
            "avg_sio2": {"$avg": "$composition.SiO2"}
        }
    },
    {"$sort": {"avg_sio2": -1}}
]

results = await storage.db.geochemical_samples.aggregate(pipeline).to_list(None)
```

## Performance Considerations

### Indexing Strategy
- **Compound indexes** for common query patterns
- **Partial indexes** for sparse fields
- **TTL indexes** for automatic data cleanup
- **Text indexes** for search functionality

### Query Optimization
- **Projection** to limit returned fields
- **Limit and skip** for pagination
- **Aggregation pipeline optimization**
- **Index hint usage** for complex queries

### Memory Management
- **Streaming cursors** for large result sets
- **Batch processing** for bulk operations
- **Connection pooling** for concurrent access
- **Memory-efficient aggregations**

## Monitoring and Maintenance

### Health Monitoring
- **Connection health checks**
- **Query performance monitoring**
- **Index usage statistics**
- **Storage utilization tracking**

### Data Quality Monitoring
- **Validation error tracking**
- **Data completeness metrics**
- **Quality score distributions**
- **Temporal coverage analysis**

### Maintenance Tasks
- **Index maintenance and optimization**
- **Data cleanup and archival**
- **Performance tuning**
- **Schema evolution management**

## Security Considerations

### Access Control
- **Role-based access control (RBAC)**
- **Collection-level permissions**
- **Field-level security**
- **Audit logging**

### Data Protection
- **Encryption at rest**
- **Encryption in transit**
- **Data anonymization**
- **Backup and recovery**

## Testing

The implementation includes comprehensive tests covering:
- **Unit tests** for storage operations
- **Integration tests** for API endpoints
- **Performance tests** for large datasets
- **Data validation tests**
- **Error handling tests**

Run tests with:
```bash
cd services/data-ingestion
pytest tests/test_mongodb_storage.py -v
```

## Configuration

MongoDB settings are configured through environment variables:

```bash
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=prometheus_geological
```

## Dependencies

- `motor`: Async MongoDB driver
- `pymongo`: MongoDB Python driver
- `fastapi`: Web framework
- `pydantic`: Data validation
- `structlog`: Structured logging

## Future Enhancements

1. **Sharding support** for horizontal scaling
2. **Change streams** for real-time notifications
3. **GridFS integration** for large file storage
4. **Advanced geospatial queries** (polygon intersections, etc.)
5. **Machine learning integration** for automated data classification
6. **Time-series collections** for high-frequency data
7. **Multi-tenant support** for different research groups