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
        rock_
class AdvancedSeismicCNN:
    """Convolutional Neural Network for seismic event detection and classification"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(input_channels, num_classes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def _build_model(self, input_channels: int, num_classes: int) -> nn.Module:
        """Build advanced CNN architecture with attention mechanisms"""
        
        class AttentionBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.attention = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Conv1d(channels, channels // 8, 1),
                    nn.ReLU(),
                    nn.Conv1d(channels // 8, channels, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                attention_weights = self.attention(x)
                return x * attention_weights
        
        class SeismicCNN(nn.Module):
            def __init__(self, input_channels, num_classes):
                super().__init__()
                
                # Feature extraction layers
                self.conv_blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        AttentionBlock(64),
                        nn.MaxPool1d(2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(64, 128, kernel_size=5, padding=2),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        AttentionBlock(128),
                        nn.MaxPool1d(2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(128, 256, kernel_size=3, padding=1),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        AttentionBlock(256),
                        nn.MaxPool1d(2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(256, 512, kernel_size=3, padding=1),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        AttentionBlock(512),
                        nn.AdaptiveAvgPool1d(1)
                    )
                ])
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
                
            def forward(self, x):
                for block in self.conv_blocks:
                    x = block(x)
                
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        return SeismicCNN(input_channels, num_classes).to(self.device)
    
    def detect_seismic_events(self, waveform_data: np.ndarray, 
                            sampling_rate: float = 100.0) -> Dict[str, Any]:
        """Detect and classify seismic events in waveform data"""
        
        # Preprocess data
        processed_data = self._preprocess_waveform(waveform_data, sampling_rate)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Event classification mapping
        event_types = {
            0: 'background_noise',
            1: 'local_earthquake',
            2: 'regional_earthquake',
            3: 'volcanic_tremor',
            4: 'explosion'
        }
        
        return {
            'event_type': event_types[predicted_class.item()],
            'confidence': probabilities.max().item(),
            'all_probabilities': {
                event_types[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'magnitude_estimate': self._estimate_magnitude(waveform_data),
            'arrival_time': self._detect_p_wave_arrival(waveform_data, sampling_rate),
            'frequency_content': self._analyze_frequency_content(waveform_data, sampling_rate)
        }
    
    def _preprocess_waveform(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Advanced waveform preprocessing"""
        # Bandpass filter
        from scipy import signal
        nyquist = sampling_rate / 2
        low_freq = 1.0 / nyquist
        high_freq = 25.0 / nyquist
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        
        # Normalize
        normalized_data = (filtered_data - np.mean(filtered_data, axis=-1, keepdims=True)) / \
                         (np.std(filtered_data, axis=-1, keepdims=True) + 1e-8)
        
        return normalized_data
    
    def _estimate_magnitude(self, waveform_data: np.ndarray) -> float:
        """Estimate earthquake magnitude from waveform amplitude"""
        max_amplitude = np.max(np.abs(waveform_data))
        
class AdvancedPhysicsSimulator:
    """Physics-Informed Neural Network for geological simulations"""
    
    def __init__(self, domain_size: Tuple[int, int, int] = (100, 100, 100)):
        self.domain_size = domain_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pinn_model = self._build_pinn_model()
        self.optimizer = torch.optim.Adam(self.pinn_model.parameters(), lr=1e-3)
        
    def solve_heat_equation_3d(self, thermal_conductivity: float = 2.5,
                              heat_capacity: float = 1000.0,
                              density: float = 2700.0) -> Dict[str, np.ndarray]:
        """Solve 3D heat equation with geological parameters"""
        
        # Generate spatial-temporal grid
        x = torch.linspace(0, 1, self.domain_size[0])
        y = torch.linspace(0, 1, self.domain_size[1])
        z = torch.linspace(0, 1, self.domain_size[2])
        t = torch.linspace(0, 1, 100)
        
        X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1)
        coords = coords.to(self.device)
        
        # Training loop with physics constraints
        for epoch in range(1000):
            self.optimizer.zero_grad()
            
            # Forward pass
            u_pred = self.pinn_model(coords)
            
            # Physics loss (heat equation)
            u_t = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 3]
            u_x = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 0]
            u_y = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 1]
            u_z = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 2]
            
            u_xx = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 0]
            u_yy = torch.autograd.grad(u_y.sum(), coords, create_graph=True)[0][:, 1]
            u_zz = torch.autograd.grad(u_z.sum(), coords, create_graph=True)[0][:, 2]
            
            alpha = thermal_conductivity / (density * heat_capacity)
            physics_loss = torch.mean((u_t - alpha * (u_xx + u_yy + u_zz))**2)
            
            physics_loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, Loss: {physics_loss.item():.6f}")
        
        # Generate solution
        with torch.no_grad():
            solution = self.pinn_model(coords).cpu().numpy()
            
        return {
            'temperature_field': solution[:, 0].reshape(self.domain_size + (100,)),
            'heat_flux_x': solution[:, 1].reshape(self.domain_size + (100,)),
            'heat_flux_y': solution[:, 2].reshape(self.domain_size + (100,)),
            'heat_flux_z': solution[:, 3].reshape(self.domain_size + (100,)),
            'thermal_conductivity': thermal_conductivity,
            'convergence_history': {'final_loss': physics_loss.item()}
        }

        magnitude = np.log10(max_amplitude * 1e6) - 3.0
        return max(0.0, min(9.0, magnitude))
    
    def _detect_p_wave_arrival(self, data: np.ndarray, sampling_rate: float) -> float:
        """Detect P-wave arrival time using STA/LTA algorithm"""
        # Short-term average / Long-term average
        sta_window = int(0.5 * sampling_rate)  # 0.5 seconds
        lta_window = int(10.0 * sampling_rate)  # 10 seconds
        
        energy = np.square(data)
        sta = np.convolve(energy, np.ones(sta_window)/sta_window, mode='same')
        lta = np.convolve(energy, np.ones(lta_window)/lta_window, mode='same')
        
        sta_lta_ratio = sta / (lta + 1e-8)
        trigger_threshold = 3.0
        
        trigger_indices = np.where(sta_lta_ratio > trigger_threshold)[0]
        if len(trigger_indices) > 0:
            return trigger_indices[0] / sampling_rate
        return 0.0
    
    def _analyze_frequency_content(self, data: np.ndarray, sampling_rate: float) -> Dict[str, float]:
        """Analyze frequency content of seismic signal"""
        from scipy import signal
        
        frequencies, psd = signal.welch(data, fs=sampling_rate, nperseg=1024)
        
        # Frequency bands of interest
        bands = {
            'low_freq': (0.1, 1.0),
            'mid_freq': (1.0, 10.0),
            'high_freq': (10.0, 25.0)
        }
        
        band_powers = {}
        for band_name, (low, high) in bands.items():
            band_mask = (frequencies >= low) & (frequencies <= high)
            band_powers[band_name] = np.trapz(psd[band_mask], frequencies[band_mask])
        
        # Dominant frequency
        dominant_freq = frequencies[np.argmax(psd)]
        
        return {
            'dominant_frequency': dominant_freq,
            'band_powers': band_powers,
            'spectral_centroid': np.sum(frequencies * psd) / np.sum(psd),
            'spectral_bandwidth': np.sqrt(np.sum(((frequencies - dominant_freq) ** 2) * psd) / np.sum(psd))
        }
 common_indexes + [
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
                                              
class AdvancedGeochemicalAnalyzer:
    """Advanced geochemical analysis with machine learning and thermodynamic modeling"""
    
    def __init__(self):
        self.element_database = self._load_element_database()
        self.mineral_database = self._load_mineral_database()
        self.ml_models = self._initialize_ml_models()
        
    def _load_element_database(self) -> Dict[str, Dict[str, float]]:
        """Load comprehensive element property database"""
        return {
            'Si': {'atomic_mass': 28.0855, 'ionic_radius': 0.40, 'electronegativity': 1.90, 'oxidation_states': [4, -4]},
            'Al': {'atomic_mass': 26.9815, 'ionic_radius': 0.535, 'electronegativity': 1.61, 'oxidation_states': [3]},
            'Fe': {'atomic_mass': 55.845, 'ionic_radius': 0.645, 'electronegativity': 1.83, 'oxidation_states': [2, 3]},
            'Mg': {'atomic_mass': 24.305, 'ionic_radius': 0.72, 'electronegativity': 1.31, 'oxidation_states': [2]},
            'Ca': {'atomic_mass': 40.078, 'ionic_radius': 1.00, 'electronegativity': 1.00, 'oxidation_states': [2]},
            'Na': {'atomic_mass': 22.990, 'ionic_radius': 1.02, 'electronegativity': 0.93, 'oxidation_states': [1]},
            'K': {'atomic_mass': 39.098, 'ionic_radius': 1.38, 'electronegativity': 0.82, 'oxidation_states': [1]},
            'Ti': {'atomic_mass': 47.867, 'ionic_radius': 0.605, 'electronegativity': 1.54, 'oxidation_states': [4, 3]},
            'Mn': {'atomic_mass': 54.938, 'ionic_radius': 0.67, 'electronegativity': 1.55, 'oxidation_states': [2, 3, 4, 7]},
            'P': {'atomic_mass': 30.974, 'ionic_radius': 0.44, 'electronegativity': 2.19, 'oxidation_states': [5, 3, -3]}
        }
    
    def _load_mineral_database(self) -> Dict[str, Dict[str, Any]]:
        """Load mineral composition and stability database"""
        return {
            'quartz': {
                'formula': 'SiO2',
                'composition': {'Si': 1, 'O': 2},
                'density': 2.65,
                'hardness': 7,
                'stability_temp': 573.0,  # Alpha-beta transition
                'formation_pressure': 1.0
            },
            'feldspar': {
                'formula': 'KAlSi3O8',
                'composition': {'K': 1, 'Al': 1, 'Si': 3, 'O': 8},
                'density': 2.56,
                'hardness': 6,
                'stability_temp': 1150.0,
                'formation_pressure': 1.0
            },
            'olivine': {
                'formula': '(Mg,Fe)2SiO4',
                'composition': {'Mg': 1.5, 'Fe': 0.5, 'Si': 1, 'O': 4},
                'density': 3.3,
                'hardness': 6.5,
                'stability_temp': 1890.0,
                'formation_pressure': 1.0
            },
            'pyroxene': {
                'formula': '(Ca,Mg,Fe)SiO3',
                'composition': {'Ca': 0.5, 'Mg': 0.8, 'Fe': 0.7, 'Si': 1, 'O': 3},
                'density': 3.4,
                'hardness': 6,
                'stability_temp': 1200.0,
                'formation_pressure': 1.0
            }
        }
    
    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialize machine learning models for geochemical analysis"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
        from sklearn.neural_network import MLPRegressor
        
        return {
            'composition_predictor': RandomForestRegressor(n_estimators=100, random_state=42),
            'rock_classifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'temperature_estimator': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42),
            'pressure_estimator': RandomForestRegressor(n_estimators=100, random_state=42)
        }
    
    def analyze_rock_composition(self, composition: Dict[str, float], 
                               temperature: float = 1200.0,
                               pressure: float = 1.0) -> Dict[str, Any]:
        """Comprehensive rock composition analysis"""
        
        # Normalize composition to 100%
        total = sum(composition.values())
        normalized_comp = {element: (value/total)*100 for element, value in composition.items()}
        
        # Calculate derived parameters
        analysis_results = {
            'normalized_composition': normalized_comp,
            'rock_classification': self._classify_rock_type(normalized_comp),
            'mineral_assemblage': self._predict_mineral_assemblage(normalized_comp, temperature, pressure),
            'geochemical_indices': self._calculate_geochemical_indices(normalized_comp),
            'thermodynamic_properties': self._calculate_thermodynamic_properties(normalized_comp, temperature, pressure),
            'trace_element_patterns': self._analyze_trace_elements(composition),
            'alteration_assessment': self._assess_alteration(normalized_comp),
            'petrogenetic_indicators': self._calculate_petrogenetic_indicators(normalized_comp)
        }
        
        return analysis_results
    
    def _classify_rock_type(self, composition: Dict[str, float]) -> Dict[str, Any]:
        """Classify rock type based on composition"""
        sio2 = composition.get('SiO2', 0)
        al2o3 = composition.get('Al2O3', 0)
        fe2o3 = composition.get('Fe2O3', 0) + composition.get('FeO', 0)
        mgo = composition.get('MgO', 0)
        cao = composition.get('CaO', 0)
        na2o = composition.get('Na2O', 0)
        k2o = composition.get('K2O', 0)
        
        # Total alkalis
        alkalis = na2o + k2o
        
        # Classification logic
        if sio2 > 63:
            if alkalis > 7:
                rock_type = 'alkaline_felsic'
            else:
                rock_type = 'calc_alkaline_felsic'
        elif sio2 > 52:
            if mgo > 6:
                rock_type = 'intermediate_mafic'
            else:
                rock_type = 'intermediate'
        else:
            if mgo > 12:
                rock_type = 'ultramafic'
            else:
                rock_type = 'mafic'
        
        # Tectonic setting inference
        if (k2o/na2o) > 0.5 and sio2 > 60:
            tectonic_setting = 'continental_arc'
        elif alkalis < 3 and mgo > 8:
            tectonic_setting = 'mid_ocean_ridge'
        elif alkalis > 5:
            tectonic_setting = 'within_plate'
        else:
            tectonic_setting = 'island_arc'
        
        return {
            'primary_type': rock_type,
            'silica_content': sio2,
            'alkali_content': alkalis,
            'mg_number': (mgo / (mgo + fe2o3 + 0.1)) * 100,
            'tectonic_setting': tectonic_setting,
            'confidence': 0.85  # Would be calculated from model uncertainty
        }
    
    def _predict_mineral_assemblage(self, composition: Dict[str, float], 
                                  temperature: float, pressure: float) -> Dict[str, Any]:
        """Predict stable mineral assemblage using thermodynamic modeling"""
        
        
class AdvancedSeismicCNN:
    """Convolutional Neural Network for seismic event detection and classification"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(input_channels, num_classes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        
    def detect_seismic_events(self, waveform_data: np.ndarray, 
                            sampling_rate: float = 100.0) -> Dict[str, Any]:
        """Detect and classify seismic events in waveform data"""
        
        # Preprocess data
        processed_data = self._preprocess_waveform(waveform_data, sampling_rate)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Event classification mapping
        event_types = {
            0: 'background_noise',
            1: 'local_earthquake',
            2: 'regional_earthquake',
            3: 'volcanic_tremor',
            4: 'explosion'
        }
        
        return {
            'event_type': event_types[predicted_class.item()],
            'confidence': probabilities.max().item(),
            'all_probabilities': {
                event_types[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'magnitude_estimate': self._estimate_magnitude(waveform_data),
            'arrival_time': self._detect_p_wave_arrival(waveform_data, sampling_rate)
        }

        stable_minerals = []
        mineral_proportions = {}
        
        sio2 = composition.get('SiO2', 0)
        al2o3 = composition.get('Al2O3', 0)
        fe2o3 = composition.get('Fe2O3', 0) + composition.get('FeO', 0)
        mgo = composition.get('MgO', 0)
        cao = composition.get('CaO', 0)
        
        # Quartz stability
        if sio2 > 60:
            stable_minerals.append('quartz')
            mineral_proportions['quartz'] = min(40, (sio2 - 60) * 2)
        
        # Feldspar stability
        if al2o3 > 10:
            stable_minerals.append('feldspar')
            mineral_proportions['feldspar'] = min(50, al2o3 * 3)
        
        # Olivine stability (high temperature, low silica)
        if temperature > 1000 and sio2 < 50 and mgo > 5:
            stable_minerals.append('olivine')
            mineral_proportions['olivine'] = min(30, mgo * 4)
        
        # Pyroxene stability
        if temperature > 800 and (cao > 5 or mgo > 3):
            stable_minerals.append('pyroxene')
            mineral_proportions['pyroxene'] = min(35, (cao + mgo) * 2)
        
        # Normalize proportions
        total_prop = sum(mineral_proportions.values())
        if total_prop > 0:
            mineral_proportions = {k: (v/total_prop)*100 for k, v in mineral_proportions.items()}
        
        return {
            'stable_minerals': stable_minerals,
            'mineral_proportions': mineral_proportions,
            'crystallization_sequence': self._determine_crystallization_sequence(temperature, composition),
            'phase_diagram_position': {
                'temperature': temperature,
                'pressure': pressure,
                'stability_field': 'intermediate' if 800 < temperature < 1200 else 'extreme'
            }
        }
    
    def _calculate_geochemical_indices(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Calculate various geochemical indices"""
        
        sio2 = composition.get('SiO2', 0)
        al2o3 = composition.get('Al2O3', 0)
        fe2o3 = composition.get('Fe2O3', 0) + composition.get('FeO', 0)
        mgo = composition.get('MgO', 0)
        cao = composition.get('CaO', 0)
        na2o = composition.get('Na2O', 0)
        k2o = composition.get('K2O', 0)
        tio2 = composition.get('TiO2', 0)
        
        indices = {}
        
        # Differentiation index
        indices['differentiation_index'] = sio2 + na2o + k2o
        
        # Solidification index
        indices['solidification_index'] = 100 * mgo / (mgo + fe2o3 + na2o + k2o)
        
        # Alumina saturation index
        molecular_al2o3 = al2o3 / 101.96
        molecular_cao = cao / 56.08
        molecular_na2o = na2o / 61.98
        molecular_k2o = k2o / 94.20
        
        indices['alumina_saturation'] = molecular_al2o3 / (molecular_cao + molecular_na2o + molecular_k2o)
        
        # Alkali-lime index
        indices['alkali_lime_index'] = (na2o + k2o) - cao
        
        # Iron enrichment index
        indices['iron_enrichment'] = fe2o3 / (fe2o3 + mgo)
        
        # Titanium enrichment
        indices['titanium_enrichment'] = tio2
        
        return indices
    
    def _calculate_thermodynamic_properties(self, composition: Dict[str, float],
                                          temperature: float, pressure: float) -> Dict[str, float]:
        """Calculate thermodynamic properties"""
        
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []

        # In practice, would use MELTS or similar thermodynamic software
        
        properties = {}
        
        # Estimate viscosity (simplified Arrhenius model)
        sio2 = composition.get('SiO2', 50)
        al2o3 = composition.get('Al2O3', 15)
        
        # Viscosity increases with SiO2 and Al2O3, decreases with temperature
        log_viscosity = (sio2 * 0.1 + al2o3 * 0.05) - (temperature - 1000) * 0.001
        properties['viscosity_log_pa_s'] = log_viscosity
        
        # Estimate density (simplified mixing model)
        density_components = {
            'SiO2': 2.65, 'Al2O3': 3.95, 'Fe2O3': 5.24, 'FeO': 5.87,
            'MgO': 3.58, 'CaO': 3.34, 'Na2O': 2.27, 'K2O': 2.32, 'TiO2': 4.23
        }
        
        total_density = 0
        total_weight = 0
        for oxide, weight in composition.items():
            if oxide in density_components:
                total_density += weight * density_components[oxide]
                total_weight += weight
        
        properties['density_g_cm3'] = total_density / (total_weight + 1e-8)
        
        # Estimate liquidus temperature
        
class AdvancedSeismicCNN:
    """Convolutional Neural Network for seismic event detection and classification"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(input_channels, num_classes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        
    def detect_seismic_events(self, waveform_data: np.ndarray, 
                            sampling_rate: float = 100.0) -> Dict[str, Any]:
        """Detect and classify seismic events in waveform data"""
        
        # Preprocess data
        processed_data = self._preprocess_waveform(waveform_data, sampling_rate)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Event classification mapping
        event_types = {
            0: 'background_noise',
            1: 'local_earthquake',
            2: 'regional_earthquake',
            3: 'volcanic_tremor',
            4: 'explosion'
        }
        
        return {
            'event_type': event_types[predicted_class.item()],
            'confidence': probabilities.max().item(),
            'all_probabilities': {
                event_types[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'magnitude_estimate': self._estimate_magnitude(waveform_data),
            'arrival_time': self._detect_p_wave_arrival(waveform_data, sampling_rate)
        }

        base_liquidus = 1200  # Base temperature in Celsius
        sio2_effect = (sio2 - 50) * 5  # SiO2 increases liquidus
        mgo_effect = composition.get('MgO', 0) * 10  # MgO increases liquidus
        alkali_effect = -(composition.get('Na2O', 0) + composition.get('K2O', 0)) * 8  # Alkalis decrease liquidus
        
        properties['liquidus_temperature_c'] = base_liquidus + sio2_effect + mgo_effect + alkali_effect
        
        # Heat capacity (simplified)
        properties['heat_capacity_j_mol_k'] = 80 + 0.02 * temperature  
class AdvancedPhysicsSimulator:
    """Physics-Informed Neural Network for geological simulations"""
    
    def __init__(self, domain_size: Tuple[int, int, int] = (100, 100, 100)):
        self.domain_size = domain_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pinn_model = self._build_pinn_model()
        self.optimizer = torch.optim.Adam(self.pinn_model.parameters(), lr=1e-3)
        
    def solve_heat_equation_3d(self, thermal_conductivity: float = 2.5,
                              heat_capacity: float = 1000.0,
                              density: float = 2700.0) -> Dict[str, np.ndarray]:
        """Solve 3D heat equation with geological parameters"""
        
        # Generate spatial-temporal grid
        x = torch.linspace(0, 1, self.domain_size[0])
        y = torch.linspace(0, 1, self.domain_size[1])
        z = torch.linspace(0, 1, self.domain_size[2])
        t = torch.linspace(0, 1, 100)
        
        X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1)
        coords = coords.to(self.device)
        
        # Training loop with physics constraints
        for epoch in range(1000):
            self.optimizer.zero_grad()
            
            # Forward pass
            u_pred = self.pinn_model(coords)
            
            # Physics loss (heat equation)
            u_t = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 3]
            u_x = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 0]
            u_y = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 1]
            u_z = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 2]
            
            u_xx = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 0]
            u_yy = torch.autograd.grad(u_y.sum(), coords, create_graph=True)[0][:, 1]
            u_zz = torch.autograd.grad(u_z.sum(), coords, create_graph=True)[0][:, 2]
            
            alpha = thermal_conductivity / (density * heat_capacity)
            physics_loss = torch.mean((u_t - alpha * (u_xx + u_yy + u_zz))**2)
            
            physics_loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, Loss: {physics_loss.item():.6f}")
        
        # Generate solution
        with torch.no_grad():
            solution = self.pinn_model(coords).cpu().numpy()
            
        return {
            'temperature_field': solution[:, 0].reshape(self.domain_size + (100,)),
            'heat_flux_x': solution[:, 1].reshape(self.domain_size + (100,)),
            'heat_flux_y': solution[:, 2].reshape(self.domain_size + (100,)),
            'heat_flux_z': solution[:, 3].reshape(self.domain_size + (100,)),
            'thermal_conductivity': thermal_conductivity,
            'convergence_history': {'final_loss': physics_loss.item()}
        }

        
        return properties
    
    def _analyze_trace_elements(self, composition: Dict[str, float]) -> Dict[str, Any]:
        """Analyze trace element patterns"""
        
        # Extract trace elements (typically < 1000 ppm)
        trace_elements = {}
        for element, concentration in composition.items():
            if concentration < 0.1:  # Less than 0.1 wt% = trace element
                trace_elements[element] = concentration * 10000  # Convert to ppm
        
        # Calculate trace element ratios
        ratios = {}
        if 'Rb' in trace_elements and 'Sr' in trace_elements:
            ratios['Rb_Sr'] = trace_elements['Rb'] / (trace_elements['Sr'] + 1e-8)
        
        if 'Ba' in trace_elements and 'Rb' in trace_elements:
            ratios['Ba_Rb'] = trace_elements['Ba'] / (trace_elements['Rb'] + 1e-8)
        
        if 'Zr' in trace_elements and 'Hf' in trace_elements:
            ratios['Zr_Hf'] = trace_elements['Zr'] / (trace_elements['Hf'] + 1e-8)
        
        # REE pattern analysis (if REE data available)
        ree_elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
        ree_pattern = {element: trace_elements.get(element, 0) for element in ree_elements}
        
        return {
            'trace_elements_ppm': trace_elements,
            'trace_element_ratios': ratios,
            'ree_pattern': ree_pattern,
            'enrichment_factors': self._calculate_enrichment_factors(trace_elements)
        }
    
    def _calculate_enrichment_factors(self, trace_elements: Dict[str, float]) -> Dict[str, float]:
        """Calculate enrichment factors relative to primitive mantle"""
        
        # Primitive mantle values (McDonough & Sun, 1995)
        primitive_mantle = {
            'Rb': 0.635, 'Ba': 6.989, 'Th': 0.085, 'U': 0.021, 'Nb': 0.713,
            'Ta': 0.041, 'La': 0.687, 'Ce': 1.775, 'Pb': 0.185, 'Pr': 0.276,
            'Sr': 21.1, 'Nd': 1.354, 'Sm': 0.444, 'Zr': 11.2, 'Hf': 0.309,
            'Eu': 0.168, 'Gd': 0.596, 'Tb': 0.108, 'Dy': 0.737, 'Y': 4.55,
            'Ho': 0.164, 'Er': 0.480, 'Tm': 0.074, 'Yb': 0.493, 'Lu': 0.074
        }
        
        enrichment_factors = {}
        for element, concentration in trace_elements.items():
            if element in primitive_mantle:
                enrichment_factors[element] = concentration / primitive_mantle[element]
        
        return enrichment_factors
 None,
                                              time_range: Optional[tuple] = None) -> Dict[str, Any]:
        """Aggregate geochemical composition statistics"""
        try:
            pipeline = []
            
            # Match stage
            match_conditions = {"data_type": "geochemical_sample"}
            if sample_type:
                match_conditions["
class AdvancedGeochemicalAnalyzer:
    """Advanced geochemical analysis with machine learning and thermodynamic modeling"""
    
    def __init__(self):
        self.element_database = self._load_element_database()
        self.mineral_database = self._load_mineral_database()
        self.ml_models = self._initialize_ml_models()
        
    def _load_element_database(self) -> Dict[str, Dict[str, float]]:
        """Load comprehensive element property database"""
        return {
            'Si': {'atomic_mass': 28.0855, 'ionic_radius': 0.40, 'electronegativity': 1.90, 'oxidation_states': [4, -4]},
            'Al': {'atomic_mass': 26.9815, 'ionic_radius': 0.535, 'electronegativity': 1.61, 'oxidation_states': [3]},
            'Fe': {'atomic_mass': 55.845, 'ionic_radius': 0.645, 'electronegativity': 1.83, 'oxidation_states': [2, 3]},
            'Mg': {'atomic_mass': 24.305, 'ionic_radius': 0.72, 'electronegativity': 1.31, 'oxidation_states': [2]},
            'Ca': {'atomic_mass': 40.078, 'ionic_radius': 1.00, 'electronegativity': 1.00, 'oxidation_states': [2]},
            'Na': {'atomic_mass': 22.990, 'ionic_radius': 1.02, 'electronegativity': 0.93, 'oxidation_states': [1]},
            'K': {'atomic_mass': 39.098, 'ionic_radius': 1.38, 'electronegativity': 0.82, 'oxidation_states': [1]},
            'Ti': {'atomic_mass': 47.867, 'ionic_radius': 0.605, 'electronegativity': 1.54, 'oxidation_states': [4, 3]},
            'Mn': {'atomic_mass': 54.938, 'ionic_radius': 0.67, 'electronegativity': 1.55, 'oxidation_states': [2, 3, 4, 7]},
            'P': {'atomic_mass': 30.974, 'ionic_radius': 0.44, 'electronegativity': 2.19, 'oxidation_states': [5, 3, -3]}
        }
    
    def _load_mineral_database(self) -> Dict[str, Dict[str, Any]]:
        """Load mineral composition and stability database"""
        return {
            'quartz': {
                'formula': 'SiO2',
                'composition': {'Si': 1, 'O': 2},
                'density': 2.65,
                'hardness': 7,
                'stability_temp': 573.0,  # Alpha-beta transition
                'formation_pressure': 1.0
            },
            'feldspar': {
                'formula': 'KAlSi3O8',
                'composition': {'K': 1, 'Al': 1, 'Si': 3, 'O': 8},
                'density': 2.56,
                'hardness': 6,
                'stability_temp': 1150.0,
                'formation_pressure': 1.0
            },
            'olivine': {
                'formula': '(Mg,Fe)2SiO4',
                'composition': {'Mg': 1.5, 'Fe': 0.5, 'Si': 1, 'O': 4},
                'density': 3.3,
                'hardness': 6.5,
                'stability_temp': 1890.0,
                'formation_pressure': 1.0
            },
            'pyroxene': {
                'formula': '(Ca,Mg,Fe)SiO3',
                'composition': {'Ca': 0.5, 'Mg': 0.8, 'Fe': 0.7, 'Si': 1, 'O': 3},
                'density': 3.4,
                'hardness': 6,
                'stability_temp': 1200.0,
                'formation_pressure': 1.0
            }
        }
    
    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialize machine learning models for geochemical analysis"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
        from sklearn.neural_network import MLPRegressor
        
        return {
            'composition_predictor': RandomForestRegressor(n_estimators=100, random_state=42),
            'rock_classifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'temperature_estimator': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42),
            'pressure_estimator': RandomForestRegressor(n_estimators=100, random_state=42)
        }
    
    def analyze_rock_composition(self, composition: Dict[str, float], 
                               temperature: float = 1200.0,
                               pressure: float = 1.0) -> Dict[str, Any]:
        """Comprehensive rock composition analysis"""
        
        # Normalize composition to 100%
        total = sum(composition.values())
        normalized_comp = {element: (value/total)*100 for element, value in composition.items()}
        
        # Calculate derived parameters
        analysis_results = {
            'normalized_composition': normalized_comp,
            'rock_classification': self._classify_rock_type(normalized_comp),
            'mineral_assemblage': self._predict_mineral_assemblage(normalized_comp, temperature, pressure),
            'geochemical_indices': self._calculate_geochemical_indices(normalized_comp),
            'thermodynamic_properties': self._calculate_thermodynamic_properties(normalized_comp, temperature, pressure),
            'trace_element_patterns': self._analyze_trace_elements(composition),
            'alteration_assessment': self._assess_alteration(normalized_comp),
            'petrogenetic_indicators': self._calculate_petrogenetic_indicators(normalized_comp)
        }
        
        return analysis_results
    
    def _classify_rock_type(self, composition: Dict[str, float]) -> Dict[str, Any]:
        """Classify rock type based on composition"""
        sio2 = composition.get('SiO2', 0)
        al2o3 = composition.get('Al2O3', 0)
        fe2o3 = composition.get('Fe2O3', 0) + composition.get('FeO', 0)
        mgo = composition.get('MgO', 0)
        cao = composition.get('CaO', 0)
        na2o = composition.get('Na2O', 0)
        k2o = composition.get('K2O', 0)
        
        # Total alkalis
        alkalis = na2o + k2o
        
        # Classification logic
        if sio2 > 63:
            if alkalis > 7:
                rock_type = 'alkaline_felsic'
            else:
                rock_type = 'calc_alkaline_felsic'
        elif sio2 > 52:
            if mgo > 6:
                rock_type = 'intermediate_mafic'
            else:
                rock_type = 'intermediate'
        else:
            if mgo > 12:
                rock_type = 'ultramafic'
            else:
                rock_type = 'mafic'
        
        # Tectonic setting inference
        if (k2o/na2o) > 0.5 and sio2 > 60:
            tectonic_setting = 'continental_arc'
        elif alkalis < 3 and mgo > 8:
            tectonic_setting = 'mid_ocean_ridge'
        elif alkalis > 5:
            tectonic_setting = 'within_plate'
        else:
            tectonic_setting = 'island_arc'
        
        return {
            'primary_type': rock_type,
            'silica_content': sio2,
            'alkali_content': alkalis,
            'mg_number': (mgo / (mgo + fe2o3 + 0.1)) * 100,
            'tectonic_setting': tectonic_setting,
            'confidence': 0.85  # Would be calculated from model uncertainty
        }
    
    def _predict_mineral_assemblage(self, composition: Dict[str, float], 
                                  temperature: float, pressure: float) -> Dict[str, Any]:
        """Predict stable mineral assemblage using thermodynamic modeling"""
        
        
class AdvancedSeismicCNN:
    """Convolutional Neural Network for seismic event detection and classification"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(input_channels, num_classes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        
    def detect_seismic_events(self, waveform_data: np.ndarray, 
                            sampling_rate: float = 100.0) -> Dict[str, Any]:
        """Detect and classify seismic events in waveform data"""
        
        # Preprocess data
        processed_data = self._preprocess_waveform(waveform_data, sampling_rate)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Event classification mapping
        event_types = {
            0: 'background_noise',
            1: 'local_earthquake',
            2: 'regional_earthquake',
            3: 'volcanic_tremor',
            4: 'explosion'
        }
        
        return {
            'event_type': event_types[predicted_class.item()],
            'confidence': probabilities.max().item(),
            'all_probabilities': {
                event_types[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'magnitude_estimate': self._estimate_magnitude(waveform_data),
            'arrival_time': self._detect_p_wave_arrival(waveform_data, sampling_rate)
        }

        stable_minerals = []
        mineral_proportions = {}
        
        sio2 = composition.get('SiO2', 0)
        al2o3 = composition.get('Al2O3', 0)
        fe2o3 = composition.get('Fe2O3', 0) + composition.get('FeO', 0)
        mgo = composition.get('MgO', 0)
        cao = composition.get('CaO', 0)
        
        # Quartz stability
        if sio2 > 60:
            stable_minerals.append('quartz')
            mineral_proportions['quartz'] = min(40, (sio2 - 60) * 2)
        
        # Feldspar stability
        if al2o3 > 10:
            stable_minerals.append('feldspar')
            mineral_proportions['feldspar'] = min(50, al2o3 * 3)
        
        # Olivine stability (high temperature, low silica)
        if temperature > 1000 and sio2 < 50 and mgo > 5:
            stable_minerals.append('olivine')
            mineral_proportions['olivine'] = min(30, mgo * 4)
        
        # Pyroxene stability
        if temperature > 800 and (cao > 5 or mgo > 3):
            stable_minerals.append('pyroxene')
            mineral_proportions['pyroxene'] = min(35, (cao + mgo) * 2)
        
        # Normalize proportions
        total_prop = sum(mineral_proportions.values())
        if total_prop > 0:
            mineral_proportions = {k: (v/total_prop)*100 for k, v in mineral_proportions.items()}
        
        return {
            'stable_minerals': stable_minerals,
            'mineral_proportions': mineral_proportions,
            'crystallization_sequence': self._determine_crystallization_sequence(temperature, composition),
            'phase_diagram_position': {
                'temperature': temperature,
                'pressure': pressure,
                'stability_field': 'intermediate' if 800 < temperature < 1200 else 'extreme'
            }
        }
    
    def _calculate_geochemical_indices(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Calculate various geochemical indices"""
        
        sio2 = composition.get('SiO2', 0)
        al2o3 = composition.get('Al2O3', 0)
        fe2o3 = composition.get('Fe2O3', 0) + composition.get('FeO', 0)
        mgo = composition.get('MgO', 0)
        cao = composition.get('CaO', 0)
        na2o = composition.get('Na2O', 0)
        k2o = composition.get('K2O', 0)
        tio2 = composition.get('TiO2', 0)
        
        indices = {}
        
        # Differentiation index
        indices['differentiation_index'] = sio2 + na2o + k2o
        
        # Solidification index
        indices['solidification_index'] = 100 * mgo / (mgo + fe2o3 + na2o + k2o)
        
        # Alumina saturation index
        molecular_al2o3 = al2o3 / 101.96
        molecular_cao = cao / 56.08
        molecular_na2o = na2o / 61.98
        molecular_k2o = k2o / 94.20
        
        indices['alumina_saturation'] = molecular_al2o3 / (molecular_cao + molecular_na2o + molecular_k2o)
        
        # Alkali-lime index
        indices['alkali_lime_index'] = (na2o + k2o) - cao
        
        # Iron enrichment index
        indices['iron_enrichment'] = fe2o3 / (fe2o3 + mgo)
        
        # Titanium enrichment
        indices['titanium_enrichment'] = tio2
        
        return indices
    
    def _calculate_thermodynamic_properties(self, composition: Dict[str, float],
                                          temperature: float, pressure: float) -> Dict[str, float]:
        """Calculate thermodynamic properties"""
        
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []

        # In practice, would use MELTS or similar thermodynamic software
        
        properties = {}
        
        # Estimate viscosity (simplified Arrhenius model)
        sio2 = composition.get('SiO2', 50)
        al2o3 = composition.get('Al2O3', 15)
        
        # Viscosity increases with SiO2 and Al2O3, decreases with temperature
        log_viscosity = (sio2 * 0.1 + al2o3 * 0.05) - (temperature - 1000) * 0.001
        properties['viscosity_log_pa_s'] = log_viscosity
        
        # Estimate density (simplified mixing model)
        density_components = {
            'SiO2': 2.65, 'Al2O3': 3.95, 'Fe2O3': 5.24, 'FeO': 5.87,
            'MgO': 3.58, 'CaO': 3.34, 'Na2O': 2.27, 'K2O': 2.32, 'TiO2': 4.23
        }
        
        total_density = 0
        total_weight = 0
        for oxide, weight in composition.items():
            if oxide in density_components:
                total_density += weight * density_components[oxide]
                total_weight += weight
        
        properties['density_g_cm3'] = total_density / (total_weight + 1e-8)
        
        # Estimate liquidus temperature
        
class AdvancedSeismicCNN:
    """Convolutional Neural Network for seismic event detection and classification"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(input_channels, num_classes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        
    def detect_seismic_events(self, waveform_data: np.ndarray, 
                            sampling_rate: float = 100.0) -> Dict[str, Any]:
        """Detect and classify seismic events in waveform data"""
        
        # Preprocess data
        processed_data = self._preprocess_waveform(waveform_data, sampling_rate)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Event classification mapping
        event_types = {
            0: 'background_noise',
            1: 'local_earthquake',
            2: 'regional_earthquake',
            3: 'volcanic_tremor',
            4: 'explosion'
        }
        
        return {
            'event_type': event_types[predicted_class.item()],
            'confidence': probabilities.max().item(),
            'all_probabilities': {
                event_types[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'magnitude_estimate': self._estimate_magnitude(waveform_data),
            'arrival_time': self._detect_p_wave_arrival(waveform_data, sampling_rate)
        }

        base_liquidus = 1200  # Base temperature in Celsius
        sio2_effect = (sio2 - 50) * 5  # SiO2 increases liquidus
        mgo_effect = composition.get('MgO', 0) * 10  # MgO increases liquidus
        alkali_effect = -(composition.get('Na2O', 0) + composition.get('K2O', 0)) * 8  # Alkalis decrease liquidus
        
        properties['liquidus_temperature_c'] = base_liquidus + sio2_effect + mgo_effect + alkali_effect
        
        # Heat capacity (simplified)
        properties['heat_capacity_j_mol_k'] = 80 + 0.02 * temperature  
class AdvancedPhysicsSimulator:
    """Physics-Informed Neural Network for geological simulations"""
    
    def __init__(self, domain_size: Tuple[int, int, int] = (100, 100, 100)):
        self.domain_size = domain_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pinn_model = self._build_pinn_model()
        self.optimizer = torch.optim.Adam(self.pinn_model.parameters(), lr=1e-3)
        
    def solve_heat_equation_3d(self, thermal_conductivity: float = 2.5,
                              heat_capacity: float = 1000.0,
                              density: float = 2700.0) -> Dict[str, np.ndarray]:
        """Solve 3D heat equation with geological parameters"""
        
        # Generate spatial-temporal grid
        x = torch.linspace(0, 1, self.domain_size[0])
        y = torch.linspace(0, 1, self.domain_size[1])
        z = torch.linspace(0, 1, self.domain_size[2])
        t = torch.linspace(0, 1, 100)
        
        X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1)
        coords = coords.to(self.device)
        
        # Training loop with physics constraints
        for epoch in range(1000):
            self.optimizer.zero_grad()
            
            # Forward pass
            u_pred = self.pinn_model(coords)
            
            # Physics loss (heat equation)
            u_t = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 3]
            u_x = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 0]
            u_y = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 1]
            u_z = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 2]
            
            u_xx = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 0]
            u_yy = torch.autograd.grad(u_y.sum(), coords, create_graph=True)[0][:, 1]
            u_zz = torch.autograd.grad(u_z.sum(), coords, create_graph=True)[0][:, 2]
            
            alpha = thermal_conductivity / (density * heat_capacity)
            physics_loss = torch.mean((u_t - alpha * (u_xx + u_yy + u_zz))**2)
            
            physics_loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, Loss: {physics_loss.item():.6f}")
        
        # Generate solution
        with torch.no_grad():
            solution = self.pinn_model(coords).cpu().numpy()
            
        return {
            'temperature_field': solution[:, 0].reshape(self.domain_size + (100,)),
            'heat_flux_x': solution[:, 1].reshape(self.domain_size + (100,)),
            'heat_flux_y': solution[:, 2].reshape(self.domain_size + (100,)),
            'heat_flux_z': solution[:, 3].reshape(self.domain_size + (100,)),
            'thermal_conductivity': thermal_conductivity,
            'convergence_history': {'final_loss': physics_loss.item()}
        }

        
        return properties
    
    def _analyze_trace_elements(self, composition: Dict[str, float]) -> Dict[str, Any]:
        """Analyze trace element patterns"""
        
        # Extract trace elements (typically < 1000 ppm)
        trace_elements = {}
        for element, concentration in composition.items():
            if concentration < 0.1:  # Less than 0.1 wt% = trace element
                trace_elements[element] = concentration * 10000  # Convert to ppm
        
        # Calculate trace element ratios
        ratios = {}
        if 'Rb' in trace_elements and 'Sr' in trace_elements:
            ratios['Rb_Sr'] = trace_elements['Rb'] / (trace_elements['Sr'] + 1e-8)
        
        if 'Ba' in trace_elements and 'Rb' in trace_elements:
            ratios['Ba_Rb'] = trace_elements['Ba'] / (trace_elements['Rb'] + 1e-8)
        
        if 'Zr' in trace_elements and 'Hf' in trace_elements:
            ratios['Zr_Hf'] = trace_elements['Zr'] / (trace_elements['Hf'] + 1e-8)
        
        # REE pattern analysis (if REE data available)
        ree_elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
        ree_pattern = {element: trace_elements.get(element, 0) for element in ree_elements}
        
        return {
            'trace_elements_ppm': trace_elements,
            'trace_element_ratios': ratios,
            'ree_pattern': ree_pattern,
            'enrichment_factors': self._calculate_enrichment_factors(trace_elements)
        }
    
    def _calculate_enrichment_factors(self, trace_elements: Dict[str, float]) -> Dict[str, float]:
        """Calculate enrichment factors relative to primitive mantle"""
        
        # Primitive mantle values (McDonough & Sun, 1995)
        primitive_mantle = {
            'Rb': 0.635, 'Ba': 6.989, 'Th': 0.085, 'U': 0.021, 'Nb': 0.713,
            'Ta': 0.041, 'La': 0.687, 'Ce': 1.775, 'Pb': 0.185, 'Pr': 0.276,
            'Sr': 21.1, 'Nd': 1.354, 'Sm': 0.444, 'Zr': 11.2, 'Hf': 0.309,
            'Eu': 0.168, 'Gd': 0.596, 'Tb': 0.108, 'Dy': 0.737, 'Y': 4.55,
            'Ho': 0.164, 'Er': 0.480, 'Tm': 0.074, 'Yb': 0.493, 'Lu': 0.074
        }
        
        enrichment_factors = {}
        for element, concentration in trace_elements.items():
            if element in primitive_mantle:
                enrichment_factors[element] = concentration / primitive_mantle[element]
        
        return enrichment_factors
 sample_type
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
                
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def __init__(self):
        self.processors = {
            'time_series': TimeSeriesProcessor(),
            'spatial': SpatialDataProcessor(),
            'spectral': SpectralAnalysisProcessor(),
            'statistical': StatisticalProcessor(),
            'ml_preprocessing': MLPreprocessor()
        }
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', str(uuid.uuid4())),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps based on configuration
        for step_name, step_config in processing_config.items():
            if step_name in self.processors:
                try:
                    processor = self.processors[step_name]
                    processed_data, step_results = processor.process(data, step_config)
                    
                    data = processed_data
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'success',
                        'results': step_results,
                        'output_size': len(data) if isinstance(data, list) else 'N/A'
                    })
                    
                except Exception as e:
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results

class TimeSeriesProcessor:
    """Advanced time series processing for geological data"""
    
    def process(self, data: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Process time series data"""
        
        if not isinstance(data, (list, np.ndarray)):
            return data, {'error': 'Invalid data type for time series processing'}
        
        data_array = np.array(data)
        results = {}
        
        # Detrending
        if config.get('detrend', False):
            from scipy import signal
            data_array = signal.detrend(data_array, type=config.get('detrend_type', 'linear'))
            results['detrended'] = True
        
        # Filtering
        if 'filter' in config:
            filter_config = config['filter']
            data_array = self._apply_filter(data_array, filter_config)
            results['filtered'] = filter_config
        
        # Resampling
        if 'resample' in config:
            re
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []
 config['resample']
            data_array = self._resample_data(data_array, resample_config)
            results['resampled'] = resample_config
        
        # Anomaly detection
        if config.get('detect_anomalies', False):
            anomalies = self._detect_anomalies(data_array, config.get('anomaly_config', {}))
            results['anomalies'] = anomalies
        
        return data_array.tolist(), results
    
    def _apply_filter(self, data: np.ndarray, filter_config: Dict[str, Any]) -> np.ndarray:
        """Apply digital filter to data"""
        from scipy import signal
        
        filter_type = filter_config.get('type', 'bandpass')
        sampling_rate = filter_config.get('sampling_rate', 100.0)
        
        if filter_type == 'bandpass':
            low_freq = filter_config.get('low_freq', 1.0)
            high_freq = filter_config.get('high_freq', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                [low_freq/nyquist, high_freq/nyquist],
                btype='band'
            )
            return signal.filtfilt(b, a, data)
        
        elif filter_type == 'lowpass':
            cutoff = filter_config.get('cutoff', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                cutoff/nyquist,
                btype='low'
            )
            return signal.filtfilt(b, a, data)
        
        return data
    
    def _resample_data(self, data: np.ndarray, resample_config: Dict[str, Any]) -> np.ndarray:
        """Resample data to different sampling rate"""
        from scipy import signal
        
        original_rate = resample_config.get('original_rate', 100.0)
        target_rate = resample_config.get('target_rate', 50.0)
        
        if original_rate != target_rate:
            num_samples = int(len(data) * target_rate / original_rate)
            return signal.resample(data, num_samples)
        
        return data
    
    def _detect_anomalies(self, data: np.ndarray, anomaly_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        
        method = anomaly_config.get('method', 'statistical')
        
        if method == 'statistical':
            # Statistical outlier detection
            mean_val = np.mean(data)
            std_val = np.std(data)
            threshold = anomaly_config.get('threshold', 3.0)
            
            anomaly_indices = np.where(np.abs(data - mean_val) > threshold * std_val)[0]
            
            return {
                'method': 'statistical',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'threshold_used': threshold
            }
        
        elif method == 'isolation_forest':
            # Machine learning based anomaly detection
            from sklearn.ensemble import IsolationForest
            
            contamination = anomaly_config.get('contamination', 0.1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            
            data_reshaped = data.reshape(-1, 1)
            anomaly_labels = clf.fit_predict(data_reshaped)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            return {
                'method': 'isolation_forest',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'contamination': contamination
            }
        
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []


            
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