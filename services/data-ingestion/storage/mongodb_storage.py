"""MongoDB storage implementation for unstructured geological data"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import structlog
import json
from bson import ObjectId
from bson.errors import InvalidId
import motor.motor_asyncio
from pymongo import IndexModel, ASCENDING, DESCENDING, GEO2D, TEXT
from pymongo.errors import DuplicateKeyError, BulkWriteError
import numpy as np

from models.geological_data import (
    GeochemicalSample, AtmosphericMeasurement, HydroacousticMeasurement,
    GravityMeasurement, MagneticMeasurement, DataQuality
)

logger = structlog.get_logger()

class MongoDBStorage:
    """MongoDB storage for unstructured geological data with flexible schemas"""
    
    def __init__(self, mongodb_url: str, database_name: str = "prometheus_geological"):
        self.mongodb_url = mongodb_url
        self.database_name = database_name
        self.client = None
        self.db = None
        self.is_initialized = False
        
        # Collection names for different data types
        self.collections = {
            'geochemical_samples': 'geochemical_samples',
            'atmospheric_measurements': 'atmospheric_measurements',
            'hydroacoustic_measurements': 'hydroacoustic_measurements',
            'gravity_measurements': 'gravity_measurements',
            'magnetic_measurements': 'magnetic_measurements',
            'geological_surveys': 'geological_surveys',
            'rock_samples': 'rock_samples',
            'mineral_analyses': 'mineral_analyses',
            'fluid_samples': 'fluid_samples',
            'core_samples': 'core_samples',
            'field_observations': 'field_observations',
            'laboratory_results': 'laboratory_results',
            'metadata_catalog': 'metadata_catalog'
        }
    
    async def initialize(self):
        """Initialize MongoDB connection and create indexes"""
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_url)
            self.db = self.client[self.database_name]
            
            # Test connection
            await self.client.admin.command('ping')
            
            # Create indexes for all collections
            await self._create_indexes()
            
            # Create validation schemas
            await self._create_validation_schemas()
            
            self.is_initialized = True
            logger.info("MongoDB storage initialized successfully", database=self.database_name)
            
        except Exception as e:
            logger.error("Failed to initialize MongoDB storage", error=str(e))
            raise
    
    async def _create_indexes(self):
        """Create advanced indexing strategies for complex geological queries"""
        
        # Common indexes for all collections
        common_indexes = [
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("location.coordinates", GEO2D)]),
            IndexModel([("source", ASCENDING)]),
            IndexModel([("quality", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("timestamp", DESCENDING), ("location.coordinates", GEO2D)]),
            IndexModel([("data_type", ASCENDING), ("timestamp", DESCENDING)])
        ]
        
        # Geochemical samples specific indexes
        geochemical_indexes = common_indexes + [
            IndexModel([("sample_type", ASCENDING)]),
            IndexModel([("composition", ASCENDING)]),
            IndexModel([("temperature", ASCENDING)]),
            IndexModel([("pressure", ASCENDING)]),
            IndexModel([("ph", ASCENDING)]),
            IndexModel([("sample_type", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("composition.SiO2", ASCENDING)]),
            IndexModel([("composition.Al2O3", ASCENDING)]),
            IndexModel([("composition.FeO", ASCENDING)]),
            IndexModel([("composition.MgO", ASCENDING)]),
            IndexModel([("composition.CaO", ASCENDING)]),
            IndexModel([("isotope_ratios.Sr87_Sr86", ASCENDING)]),
            IndexModel([("isotope_ratios.Nd143_Nd144", ASCENDING)]),
            IndexModel([("isotope_ratios.Pb206_Pb204", ASCENDING)])
        ]
        
        # Atmospheric measurements specific indexes
        atmospheric_indexes = common_indexes + [
            IndexModel([("pressure", ASCENDING)]),
            IndexModel([("wind_speed", ASCENDING)]),
            IndexModel([("wind_direction", ASCENDING)]),
            IndexModel([("gas_concentrations.CO2", ASCENDING)]),
            IndexModel([("gas_concentrations.SO2", ASCENDING)]),
            IndexModel([("gas_concentrations.H2S", ASCENDING)]),
            IndexModel([("location.elevation", ASCENDING)])
        ]
        
        # Gravity measurements specific indexes
        gravity_indexes = common_indexes + [
            IndexModel([("gravity_value", ASCENDING)]),
            IndexModel([("measurement_type", ASCENDING)]),
            IndexModel([("uncertainty", ASCENDING)]),
            IndexModel([("reference_datum", ASCENDING)])
        ]
        
        # Magnetic measurements specific indexes
        magnetic_indexes = common_indexes + [
            IndexModel([("total_field", ASCENDING)]),
            IndexModel([("declination", ASCENDING)]),
            IndexModel([("inclination", ASCENDING)]),
            IndexModel([("measurement_type", ASCENDING)])
        ]
        
        # Rock samples specific indexes
        rock_sample_indexes = common_indexes + [
            IndexModel([("rock_type", ASCENDING)]),
            IndexModel([("formation", ASCENDING)]),
            IndexModel([("age_ma", ASCENDING)]),
            IndexModel([("texture", ASCENDING)]),
            IndexModel([("mineral_assemblage", ASCENDING)]),
            IndexModel([("metamorphic_grade", ASCENDING)]),
            IndexModel([("alteration_type", ASCENDING)])
        ]
        
        # Mineral analyses specific indexes
        mineral_indexes = common_indexes + [
            IndexModel([("mineral_name", ASCENDING)]),
            IndexModel([("crystal_system", ASCENDING)]),
            IndexModel([("chemical_formula", ASCENDING)]),
            IndexModel([("hardness", ASCENDING)]),
            IndexModel([("specific_gravity", ASCENDING)]),
            IndexModel([("optical_properties.refractive_index", ASCENDING)])
        ]
        
        # Text search indexes for flexible querying
        text_indexes = [
            IndexModel([("$**", TEXT)])  # Full-text search on all fields
        ]
        
        # Create indexes for each collection
        collections_indexes = {
            'geochemical_samples': geochemical_indexes + text_indexes,
            'atmospheric_measurements': atmospheric_indexes + text_indexes,
            'gravity_measurements': gravity_indexes + text_indexes,
            'magnetic_measurements': magnetic_indexes + text_indexes,
            'rock_samples': rock_sample_indexes + text_indexes,
            'mineral_analyses': mineral_indexes + text_indexes,
            'hydroacoustic_measurements': common_indexes + text_indexes,
            'geological_surveys': common_indexes + text_indexes,
            'fluid_samples': common_indexes + text_indexes,
            'core_samples': common_indexes + text_indexes,
            'field_observations': common_indexes + text_indexes,
            'laboratory_results': common_indexes + text_indexes,
            'metadata_catalog': common_indexes + text_indexes
        }
        
        # Create indexes for all collections
        for collection_name, indexes in collections_indexes.items():
            try:
                collection = self.db[collection_name]
                await collection.create_indexes(indexes)
                logger.debug(f"Created indexes for {collection_name}", count=len(indexes))
            except Exception as e:
                logger.warning(f"Failed to create indexes for {collection_name}", error=str(e))
    
    async def _create_validation_schemas(self):
        """Create validation schemas for data quality control"""
        
        # Geochemical sample validation schema
        geochemical_schema = {
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
                    "sample_type": {"bsonType": "string"},
                    "data_type": {"bsonType": "string"},
                    "quality": {"enum": ["excellent", "good", "fair", "poor", "unknown"]},
                    "temperature": {"bsonType": ["double", "null"], "minimum": 0, "maximum": 2000},
                    "pressure": {"bsonType": ["double", "null"], "minimum": 0},
                    "ph": {"bsonType": ["double", "null"], "minimum": 0, "maximum": 14},
                    "composition": {"bsonType": ["object", "null"]},
                    "isotope_ratios": {"bsonType": ["object", "null"]}
                }
            }
        }
        
        # Atmospheric measurement validation schema
        atmospheric_schema = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["timestamp", "location", "data_type"],
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
                    "data_type": {"bsonType": "string"},
                    "quality": {"enum": ["excellent", "good", "fair", "poor", "unknown"]},
                    "pressure": {"bsonType": ["double", "null"], "minimum": 0},
                    "temperature": {"bsonType": ["double", "null"], "minimum": 0, "maximum": 400},
                    "humidity": {"bsonType": ["double", "null"], "minimum": 0, "maximum": 100},
                    "wind_speed": {"bsonType": ["double", "null"], "minimum": 0},
                    "wind_direction": {"bsonType": ["double", "null"], "minimum": 0, "maximum": 360}
                }
            }
        }
        
        # Apply validation schemas
        validation_schemas = {
            'geochemical_samples': geochemical_schema,
            'atmospheric_measurements': atmospheric_schema
        }
        
        for collection_name, schema in validation_schemas.items():
            try:
                await self.db.create_collection(
                    collection_name,
                    validator=schema,
                    validationLevel="moderate",
                    validationAction="warn"
                )
                logger.debug(f"Created validation schema for {collection_name}")
            except Exception as e:
                # Collection might already exist
                logger.debug(f"Validation schema for {collection_name} already exists or failed", error=str(e))
    
    def _convert_to_geojson_location(self, location) -> Dict[str, Any]:
        """Convert location to GeoJSON format for MongoDB"""
        coordinates = [location.longitude, location.latitude]
        if location.elevation is not None:
            coordinates.append(location.elevation)
        
        return {
            "type": "Point",
            "coordinates": coordinates
        }
    
    def _validate_data_quality(self, data: Dict[str, Any]) -> List[str]:
        """Validate data quality and return list of issues"""
        issues = []
        
        # Check required fields
        required_fields = ['timestamp', 'location', 'data_type']
        for field in required_fields:
            if field not in data or data[field] is None:
                issues.append(f"Missing required field: {field}")
        
        # Check timestamp validity
        if 'timestamp' in data:
            if not isinstance(data['timestamp'], datetime):
                issues.append("Invalid timestamp format")
            elif data['timestamp'] > datetime.utcnow() + timedelta(hours=1):
                issues.append("Timestamp is in the future")
        
        # Check location validity
        if 'location' in data and data['location']:
            location = data['location']
            if 'coordinates' in location:
                coords = location['coordinates']
                if len(coords) >= 2:
                    lon, lat = coords[0], coords[1]
                    if not (-180 <= lon <= 180):
                        issues.append(f"Invalid longitude: {lon}")
                    if not (-90 <= lat <= 90):
                        issues.append(f"Invalid latitude: {lat}")
        
        # Check numerical ranges for specific data types
        if data.get('data_type') == 'geochemical_sample':
            if 'temperature' in data and data['temperature'] is not None:
                if not (0 <= data['temperature'] <= 2000):
                    issues.append(f"Temperature out of range: {data['temperature']}")
            if 'ph' in data and data['ph'] is not None:
                if not (0 <= data['ph'] <= 14):
                    issues.append(f"pH out of range: {data['ph']}")
        
        return issues
    
    async def store_geochemical_sample(self, sample: GeochemicalSample) -> str:
        """Store geochemical sample with flexible schema"""
        try:
            doc = {
                "timestamp": sample.timestamp,
                "location": self._convert_to_geojson_location(sample.location),
                "sample_type": sample.sample_type,
                "composition": sample.composition,
                "isotope_ratios": sample.isotope_ratios,
                "temperature": sample.temperature,
                "pressure": sample.pressure,
                "ph": sample.ph,
                "source": sample.source,
                "quality": sample.quality.value,
                "data_type": "geochemical_sample",
                "created_at": datetime.utcnow(),
                "metadata": {
                    "coordinate_system": sample.location.coordinate_system,
                    "depth": sample.location.depth,
                    "elevation": sample.location.elevation,
                    "uncertainty_m": sample.location.uncertainty_m
                }
            }
            
            # Validate data quality
            issues = self._validate_data_quality(doc)
            if issues:
                doc["validation_issues"] = issues
                logger.warning("Data quality issues detected", issues=issues)
            
            result = await self.db[self.collections['geochemical_samples']].insert_one(doc)
            logger.debug("Stored geochemical sample", id=str(result.inserted_id))
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error("Failed to store geochemical sample", error=str(e))
            raise
    
    async def store_atmospheric_measurement(self, measurement: AtmosphericMeasurement) -> str:
        """Store atmospheric measurement with flexible schema"""
        try:
            doc = {
                "timestamp": measurement.timestamp,
                "location": self._convert_to_geojson_location(measurement.location),
                "pressure": measurement.pressure,
                "temperature": measurement.temperature,
                "humidity": measurement.humidity,
                "wind_speed": measurement.wind_speed,
                "wind_direction": measurement.wind_direction,
                "gas_concentrations": measurement.gas_concentrations,
                "source": measurement.source,
                "quality": measurement.quality.value,
                "data_type": "atmospheric_measurement",
                "created_at": datetime.utcnow(),
                "metadata": {
                    "coordinate_system": measurement.location.coordinate_system,
                    "depth": measurement.location.depth,
                    "elevation": measurement.location.elevation,
                    "uncertainty_m": measurement.location.uncertainty_m
                }
            }
            
            # Validate data quality
            issues = self._validate_data_quality(doc)
            if issues:
                doc["validation_issues"] = issues
                logger.warning("Data quality issues detected", issues=issues)
            
            result = await self.db[self.collections['atmospheric_measurements']].insert_one(doc)
            logger.debug("Stored atmospheric measurement", id=str(result.inserted_id))
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error("Failed to store atmospheric measurement", error=str(e))
            raise
    
    async def store_unstructured_data(self, data: Dict[str, Any], collection_name: str) -> str:
        """Store arbitrary unstructured geological data"""
        try:
            # Add standard metadata
            doc = {
                **data,
                "created_at": datetime.utcnow(),
                "data_type": collection_name
            }
            
            # Convert location to GeoJSON if present
            if 'location' in data and isinstance(data['location'], dict):
                if 'latitude' in data['location'] and 'longitude' in data['location']:
                    doc['location'] = {
                        "type": "Point",
                        "coordinates": [data['location']['longitude'], data['location']['latitude']]
                    }
                    if 'elevation' in data['location']:
                        doc['location']['coordinates'].append(data['location']['elevation'])
            
            # Validate data quality
            issues = self._validate_data_quality(doc)
            if issues:
                doc["validation_issues"] = issues
                logger.warning("Data quality issues detected", issues=issues, collection=collection_name)
            
            # Use appropriate collection or default
            collection = self.collections.get(collection_name, collection_name)
            result = await self.db[collection].insert_one(doc)
            
            logger.debug("Stored unstructured data", 
                        id=str(result.inserted_id), 
                        collection=collection)
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error("Failed to store unstructured data", 
                        error=str(e), 
                        collection=collection_name)
            raise
    
    async def bulk_insert(self, documents: List[Dict[str, Any]], collection_name: str) -> List[str]:
        """Bulk insert documents with validation"""
        try:
            # Validate and prepare documents
            prepared_docs = []
            for doc in documents:
                # Add metadata
                prepared_doc = {
                    **doc,
                    "created_at": datetime.utcnow(),
                    "data_type": collection_name
                }
                
                # Validate data quality
                issues = self._validate_data_quality(prepared_doc)
                if issues:
                    prepared_doc["validation_issues"] = issues
                
                prepared_docs.append(prepared_doc)
            
            # Use appropriate collection
            collection = self.collections.get(collection_name, collection_name)
            result = await self.db[collection].insert_many(prepared_docs, ordered=False)
            
            inserted_ids = [str(id) for id in result.inserted_ids]
            logger.info("Bulk inserted documents", 
                       count=len(inserted_ids), 
                       collection=collection)
            return inserted_ids
            
        except BulkWriteError as e:
            # Handle partial success in bulk operations
            inserted_ids = [str(id) for id in e.details.get('insertedIds', [])]
            logger.warning("Bulk insert partially failed", 
                          inserted=len(inserted_ids),
                          errors=len(e.details.get('writeErrors', [])),
                          collection=collection_name)
            return inserted_ids
        except Exception as e:
            logger.error("Failed bulk insert", error=str(e), collection=collection_name)
            raise
    
    async def query_by_location(self, 
                               longitude: float, 
                               latitude: float, 
                               radius_km: float,
                               collection_name: str,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Query documents by geographic location"""
        try:
            collection = self.collections.get(collection_name, collection_name)
            
            # Convert radius from km to radians (Earth radius ≈ 6371 km)
            radius_radians = radius_km / 6371.0
            
            query = {
                "location": {
                    "$geoWithin": {
                        "$centerSphere": [[longitude, latitude], radius_radians]
                    }
                }
            }
            
            cursor = self.db[collection].find(query).limit(limit)
            results = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string for JSON serialization
            for result in results:
                result['_id'] = str(result['_id'])
            
            logger.debug("Location query completed", 
                        count=len(results), 
                        collection=collection)
            return results
            
        except Exception as e:
            logger.error("Failed location query", error=str(e))
            raise
    
    async def query_by_time_range(self,
                                 start_time: datetime,
                                 end_time: datetime,
                                 collection_name: str,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """Query documents by time range"""
        try:
            collection = self.collections.get(collection_name, collection_name)
            
            query = {
                "timestamp": {
                    "$gte": start_time,
                    "$lte": end_time
                }
            }
            
            cursor = self.db[collection].find(query).sort("timestamp", -1).limit(limit)
            results = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string
            for result in results:
                result['_id'] = str(result['_id'])
            
            logger.debug("Time range query completed", 
                        count=len(results), 
                        collection=collection)
            return results
            
        except Exception as e:
            logger.error("Failed time range query", error=str(e))
            raise
    
    async def text_search(self, 
                         search_text: str, 
                         collection_name: str,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Perform full-text search across documents"""
        try:
            collection = self.collections.get(collection_name, collection_name)
            
            query = {"$text": {"$search": search_text}}
            projection = {"score": {"$meta": "textScore"}}
            
            cursor = self.db[collection].find(
                query, projection
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            results = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string
            for result in results:
                result['_id'] = str(result['_id'])
            
            logger.debug("Text search completed", 
                        count=len(results), 
                        collection=collection,
                        query=search_text)
            return results
            
        except Exception as e:
            logger.error("Failed text search", error=str(e))
            raise
    
    async def aggregate_geochemical_composition(self, 
                                              sample_type: Optional[str] = None,
                                              time_range: Optional[tuple] = None) -> Dict[str, Any]:
        """Aggregate geochemical composition statistics"""
        try:
            pipeline = []
            
            # Match stage
            match_conditions = {"data_type": "geochemical_sample"}
            if sample_type:
                match_conditions["sample_type"] = sample_type
            if time_range:
                match_conditions["timestamp"] = {
                    "$gte": time_range[0],
                    "$lte": time_range[1]
                }
            
            pipeline.append({"$match": match_conditions})
            
            # Unwind composition data
            pipeline.append({"$unwind": {"path": "$composition", "preserveNullAndEmptyArrays": True}})
            
            # Group and calculate statistics
            pipeline.append({
                "$group": {
                    "_id": "$composition.k",
                    "count": {"$sum": 1},
                    "avg_concentration": {"$avg": "$composition.v"},
                    "min_concentration": {"$min": "$composition.v"},
                    "max_concentration": {"$max": "$composition.v"},
                    "std_dev": {"$stdDevPop": "$composition.v"}
                }
            })
            
            # Sort by average concentration
            pipeline.append({"$sort": {"avg_concentration": -1}})
            
            cursor = self.db[self.collections['geochemical_samples']].aggregate(pipeline)
            results = await cursor.to_list(length=None)
            
            logger.debug("Geochemical composition aggregation completed", 
                        elements=len(results))
            return {"composition_statistics": results}
            
        except Exception as e:
            logger.error("Failed geochemical composition aggregation", error=str(e))
            raise
    
    async def aggregate_temporal_trends(self, 
                                       collection_name: str,
                                       field_name: str,
                                       time_interval: str = "day") -> Dict[str, Any]:
        """Aggregate temporal trends for a specific field"""
        try:
            collection = self.collections.get(collection_name, collection_name)
            
            # Define time grouping based on interval
            time_groupings = {
                "hour": {
                    "year": {"$year": "$timestamp"},
                    "month": {"$month": "$timestamp"},
                    "day": {"$dayOfMonth": "$timestamp"},
                    "hour": {"$hour": "$timestamp"}
                },
                "day": {
                    "year": {"$year": "$timestamp"},
                    "month": {"$month": "$timestamp"},
                    "day": {"$dayOfMonth": "$timestamp"}
                },
                "month": {
                    "year": {"$year": "$timestamp"},
                    "month": {"$month": "$timestamp"}
                },
                "year": {
                    "year": {"$year": "$timestamp"}
                }
            }
            
            pipeline = [
                {"$match": {field_name: {"$exists": True, "$ne": None}}},
                {
                    "$group": {
                        "_id": time_groupings.get(time_interval, time_groupings["day"]),
                        "count": {"$sum": 1},
                        "avg_value": {"$avg": f"${field_name}"},
                        "min_value": {"$min": f"${field_name}"},
                        "max_value": {"$max": f"${field_name}"},
                        "std_dev": {"$stdDevPop": f"${field_name}"}
                    }
                },
                {"$sort": {"_id": 1}}
            ]
            
            cursor = self.db[collection].aggregate(pipeline)
            results = await cursor.to_list(length=None)
            
            logger.debug("Temporal trends aggregation completed", 
                        collection=collection,
                        field=field_name,
                        intervals=len(results))
            return {"temporal_trends": results}
            
        except Exception as e:
            logger.error("Failed temporal trends aggregation", error=str(e))
            raise
    
    async def aggregate_spatial_distribution(self, 
                                           collection_name: str,
                                           grid_size: float = 0.1) -> Dict[str, Any]:
        """Aggregate spatial distribution of data points"""
        try:
            collection = self.collections.get(collection_name, collection_name)
            
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
                        "_id": {
                            "lon": "$grid_lon",
                            "lat": "$grid_lat"
                        },
                        "count": {"$sum": 1},
                        "data_types": {"$addToSet": "$data_type"},
                        "quality_distribution": {"$push": "$quality"}
                    }
                },
                {"$sort": {"count": -1}}
            ]
            
            cursor = self.db[collection].aggregate(pipeline)
            results = await cursor.to_list(length=None)
            
            logger.debug("Spatial distribution aggregation completed", 
                        collection=collection,
                        grid_cells=len(results))
            return {"spatial_distribution": results}
            
        except Exception as e:
            logger.error("Failed spatial distribution aggregation", error=str(e))
            raise
    
    async def get_data_quality_report(self, collection_name: str) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        try:
            collection = self.collections.get(collection_name, collection_name)
            
            pipeline = [
                {
                    "$facet": {
                        "total_count": [{"$count": "count"}],
                        "quality_distribution": [
                            {"$group": {"_id": "$quality", "count": {"$sum": 1}}},
                            {"$sort": {"count": -1}}
                        ],
                        "validation_issues": [
                            {"$match": {"validation_issues": {"$exists": True}}},
                            {"$unwind": "$validation_issues"},
                            {"$group": {"_id": "$validation_issues", "count": {"$sum": 1}}},
                            {"$sort": {"count": -1}}
                        ],
                        "temporal_coverage": [
                            {
                                "$group": {
                                    "_id": None,
                                    "earliest": {"$min": "$timestamp"},
                                    "latest": {"$max": "$timestamp"},
                                    "count": {"$sum": 1}
                                }
                            }
                        ],
                        "source_distribution": [
                            {"$group": {"_id": "$source", "count": {"$sum": 1}}},
                            {"$sort": {"count": -1}}
                        ]
                    }
                }
            ]
            
            cursor = self.db[collection].aggregate(pipeline)
            results = await cursor.to_list(length=1)
            
            if results:
                report = results[0]
                logger.debug("Data quality report generated", collection=collection)
                return report
            else:
                return {"error": "No data found"}
            
        except Exception as e:
            logger.error("Failed to generate data quality report", error=str(e))
            raise
    
    async def cleanup_old_data(self, 
                              collection_name: str, 
                              days_to_keep: int = 365) -> int:
        """Clean up old data based on retention policy"""
        try:
            collection = self.collections.get(collection_name, collection_name)
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            result = await self.db[collection].delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            
            deleted_count = result.deleted_count
            logger.info("Cleaned up old data", 
                       collection=collection,
                       deleted_count=deleted_count,
                       cutoff_date=cutoff_date)
            return deleted_count
            
        except Exception as e:
            logger.error("Failed to cleanup old data", error=str(e))
            raise
    
    async def is_healthy(self) -> bool:
        """Check MongoDB connection health"""
        try:
            if self.client:
                await self.client.admin.command('ping')
                return True
            return False
        except Exception as e:
            logger.error("MongoDB health check failed", error=str(e))
            return False
    
    async def cleanup(self):
        """Cleanup MongoDB connection"""
        try:
            if self.client:
                self.client.close()
            logger.info("MongoDB storage cleanup completed")
        except Exception as e:
            logger.error("Error during MongoDB cleanup", error=str(e))