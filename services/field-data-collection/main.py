#!/usr/bin/env python3
"""
Prometheus AI Platform - Mobile Field Data Collection Service
Advanced mobile and field data collection with offline capabilities
"""

import asyncio
import json
import logging
import uuid
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import os
from pathlib import Path

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, validator
import jwt
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, Float, LargeBinary, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
import aiofiles
from PIL import Image, ExifTags
import qrcode
from pyzbar import pyzbar
import cv2
import numpy as np

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Database Models
Base = declarative_base()

class FieldSurvey(Base):
    __tablename__ = "field_surveys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    survey_name = Column(String(255), nullable=False)
    description = Column(Text)
    created_by = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    location_bounds = Column(JSONB)  # Geographic bounds
    survey_type = Column(String(50))  # geological, geophysical, geochemical
    status = Column(String(20), default="active")  # active, completed, archived
    metadata = Column(JSONB)

class FieldObservation(Base):
    __tablename__ = "field_observations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    survey_id = Column(UUID(as_uuid=True), ForeignKey("field_surveys.id"))
    observer_id = Column(UUID(as_uuid=True), nullable=False)
    observation_type = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    latitude = Column(Float)
    longitude = Column(Float)
    elevation = Column(Float)
    accuracy = Column(Float)  # GPS accuracy in meters
    description = Column(Text)
    measurements = Column(JSONB)
    photos = Column(JSONB)  # Photo metadata and paths
    audio_notes = Column(JSONB)  # Audio note metadata
    weather_conditions = Column(JSONB)
    equipment_used = Column(JSONB)
    quality_score = Column(Float, default=1.0)
    is_validated = Column(Boolean, default=False)
    sync_status = Column(String(20), default="pending")  # pending, synced, conflict

class FieldPhoto(Base):
    __tablename__ = "field_photos"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    observation_id = Column(UUID(as_uuid=True), ForeignKey("field_observations.id"))
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    file_path = Column(String(500))
    file_size = Column(Integer)
    mime_type = Column(String(100))
    width = Column(Integer)
    height = Column(Integer)
    taken_at = Column(DateTime)
    latitude = Column(Float)
    longitude = Column(Float)
    camera_info = Column(JSONB)
    exif_data = Column(JSONB)
    tags = Column(JSONB)
    processed = Column(Boolean, default=False)
    thumbnail_path = Column(String(500))

class FieldAudioNote(Base):
    __tablename__ = "field_audio_notes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    observation_id = Column(UUID(as_uuid=True), ForeignKey("field_observations.id"))
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500))
    duration_seconds = Column(Float)
    file_size = Column(Integer)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    transcription = Column(Text)
    transcription_confidence = Column(Float)
    processed = Column(Boolean, default=False)

class FieldMeasurement(Base):
    __tablename__ = "field_measurements"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    observation_id = Column(UUID(as_uuid=True), ForeignKey("field_observations.id"))
    measurement_type = Column(String(50), nullable=False)  # strike, dip, magnetic, gravity, etc.
    value = Column(Float)
    unit = Column(String(20))
    instrument = Column(String(100))
    accuracy = Column(Float)
    calibration_date = Column(DateTime)
    environmental_conditions = Column(JSONB)
    notes = Column(Text)

class OfflineDataQueue(Base):
    __tablename__ = "offline_data_queue"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    data_type = Column(String(50), nullable=False)
    data_payload = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    sync_attempts = Column(Integer, default=0)
    last_sync_attempt = Column(DateTime)
    sync_status = Column(String(20), default="pending")
    error_message = Column(Text)

# Pydantic Models
class GPSCoordinate(BaseModel):
    """GPS coordinate with accuracy"""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    elevation: Optional[float] = None
    accuracy: Optional[float] = None  # meters
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class WeatherConditions(BaseModel):
    """Weather conditions during observation"""
    temperature: Optional[float] = None  # Celsius
    humidity: Optional[float] = None  # percentage
    wind_speed: Optional[float] = None  # m/s
    wind_direction: Optional[float] = None  # degrees
    visibility: Optional[str] = None  # clear, cloudy, foggy, etc.
    precipitation: Optional[str] = None  # none, light, heavy

class FieldMeasurementData(BaseModel):
    """Field measurement data"""
    measurement_type: str
    value: float
    unit: str
    instrument: Optional[str] = None
    accuracy: Optional[float] = None
    notes: Optional[str] = None

class CreateObservationRequest(BaseModel):
    """Request to create field observation"""
    survey_id: str
    observation_type: str
    description: str
    location: GPSCoordinate
    measurements: List[FieldMeasurementData] = Field(default_factory=list)
    weather: Optional[WeatherConditions] = None
    equipment_used: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class ObservationResponse(BaseModel):
    """Field observation response"""
    id: str
    survey_id: str
    observation_type: str
    description: str
    timestamp: datetime
    location: GPSCoordinate
    measurements: List[Dict[str, Any]]
    photos: List[Dict[str, Any]]
    audio_notes: List[Dict[str, Any]]
    weather: Optional[WeatherConditions]
    quality_score: float
    is_validated: bool
    sync_status: str

class SurveyResponse(BaseModel):
    """Field survey response"""
    id: str
    survey_name: str
    description: Optional[str]
    created_by: str
    created_at: datetime
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    survey_type: str
    status: str
    observation_count: int
    location_bounds: Optional[Dict[str, Any]]

class QRCodeData(BaseModel):
    """QR code data structure"""
    type: str  # sample, location, equipment
    id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BarcodeData(BaseModel):
    """Barcode data structure"""
    format: str  # CODE128, QR_CODE, etc.
    data: str
    confidence: float

# Enums
class ObservationType(Enum):
    GEOLOGICAL = "geological"
    STRUCTURAL = "structural"
    GEOCHEMICAL = "geochemical"
    GEOPHYSICAL = "geophysical"
    ENVIRONMENTAL = "environmental"
    SAMPLE = "sample"

class SyncStatus(Enum):
    PENDING = "pending"
    SYNCED = "synced"
    CONFLICT = "conflict"
    ERROR = "error"

# Advanced Field Data Collection Engine
class FieldDataCollectionEngine:
    """Advanced field data collection and management system"""
    
    def __init__(self, redis_client: redis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.storage_path = Path("/app/data/field_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    async def create_observation(self, user_id: str, request: CreateObservationRequest) -> FieldObservation:
        """Create new field observation"""
        try:
            observation = FieldObservation(
                survey_id=uuid.UUID(request.survey_id),
                observer_id=uuid.UUID(user_id),
                observation_type=request.observation_type,
                latitude=request.location.latitude,
                longitude=request.location.longitude,
                elevation=request.location.elevation,
                accuracy=request.location.accuracy,
                description=request.description,
                measurements=[m.dict() for m in request.measurements],
                weather_conditions=request.weather.dict() if request.weather else None,
                equipment_used=request.equipment_used
            )
            
            self.db.add(observation)
            await self.db.commit()
            await self.db.refresh(observation)
            
            # Calculate quality score
            quality_score = await self._calculate_quality_score(observation)
            observation.quality_score = quality_score
            await self.db.commit()
            
            logger.info("Field observation created", 
                       observation_id=str(observation.id),
                       survey_id=request.survey_id)
            
            return observation
            
        except Exception as e:
            logger.error("Failed to create observation", error=str(e))
            raise
    
    async def process_photo_upload(self, observation_id: str, file: UploadFile) -> FieldPhoto:
        """Process and store field photo"""
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = Path(file.filename).suffix.lower()
            filename = f"{file_id}{file_extension}"
            
            # Create directory structure
            photo_dir = self.storage_path / "photos" / observation_id
            photo_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = photo_dir / filename
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Process image and extract metadata
            image_metadata = await self._process_image(file_path, content)
            
            # Create database record
            photo = FieldPhoto(
                observation_id=uuid.UUID(observation_id),
                filename=filename,
                original_filename=file.filename,
                file_path=str(file_path),
                file_size=len(content),
                mime_type=file.content_type,
                width=image_metadata.get("width"),
                height=image_metadata.get("height"),
                taken_at=image_metadata.get("taken_at"),
                latitude=image_metadata.get("latitude"),
                longitude=image_metadata.get("longitude"),
                camera_info=image_metadata.get("camera_info"),
                exif_data=image_metadata.get("exif_data")
            )
            
            self.db.add(photo)
            await self.db.commit()
            await self.db.refresh(photo)
            
            # Generate thumbnail
            await self._generate_thumbnail(photo)
            
            logger.info("Photo processed", photo_id=str(photo.id), observation_id=observation_id)
            
            return photo
            
        except Exception as e:
            logger.error("Failed to process photo", error=str(e))
            raise
    
    async def process_audio_upload(self, observation_id: str, file: UploadFile) -> FieldAudioNote:
        """Process and store audio note"""
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = Path(file.filename).suffix.lower()
            filename = f"{file_id}{file_extension}"
            
            # Create directory structure
            audio_dir = self.storage_path / "audio" / observation_id
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = audio_dir / filename
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Process audio metadata
            audio_metadata = await self._process_audio(file_path)
            
            # Create database record
            audio_note = FieldAudioNote(
                observation_id=uuid.UUID(observation_id),
                filename=filename,
                file_path=str(file_path),
                duration_seconds=audio_metadata.get("duration"),
                file_size=len(content)
            )
            
            self.db.add(audio_note)
            await self.db.commit()
            await self.db.refresh(audio_note)
            
            # Queue for transcription
            await self._queue_audio_transcription(audio_note)
            
            logger.info("Audio note processed", audio_id=str(audio_note.id), observation_id=observation_id)
            
            return audio_note
            
        except Exception as e:
            logger.error("Failed to process audio", error=str(e))
            raise
    
    async def scan_qr_code(self, image_data: bytes) -> List[QRCodeData]:
        """Scan and decode QR codes from image"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Decode QR codes
            qr_codes = pyzbar.decode(image)
            
            results = []
            for qr_code in qr_codes:
                try:
                    # Decode data
                    data = qr_code.data.decode('utf-8')
                    
                    # Try to parse as JSON (for structured QR codes)
                    try:
                        parsed_data = json.loads(data)
                        if isinstance(parsed_data, dict) and "type" in parsed_data:
                            results.append(QRCodeData(**parsed_data))
                        else:
                            # Simple text QR code
                            results.append(QRCodeData(
                                type="text",
                                id=str(uuid.uuid4()),
                                metadata={"text": data}
                            ))
                    except json.JSONDecodeError:
                        # Simple text QR code
                        results.append(QRCodeData(
                            type="text",
                            id=str(uuid.uuid4()),
                            metadata={"text": data}
                        ))
                        
                except Exception as e:
                    logger.warning("Failed to decode QR code", error=str(e))
                    continue
            
            return results
            
        except Exception as e:
            logger.error("Failed to scan QR codes", error=str(e))
            raise
    
    async def generate_qr_code(self, data: QRCodeData) -> bytes:
        """Generate QR code for data"""
        try:
            # Convert to JSON
            qr_data = json.dumps(data.dict())
            
            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(qr_data)
            qr.make(fit=True)
            
            # Create image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to bytes
            import io
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            return img_bytes.getvalue()
            
        except Exception as e:
            logger.error("Failed to generate QR code", error=str(e))
            raise
    
    async def sync_offline_data(self, user_id: str) -> Dict[str, Any]:
        """Synchronize offline data queue"""
        try:
            # Get pending offline data
            from sqlalchemy import select
            query = select(OfflineDataQueue).where(
                OfflineDataQueue.user_id == uuid.UUID(user_id),
                OfflineDataQueue.sync_status == "pending"
            )
            result = await self.db.execute(query)
            pending_items = result.scalars().all()
            
            sync_results = {
                "total_items": len(pending_items),
                "synced": 0,
                "conflicts": 0,
                "errors": 0,
                "details": []
            }
            
            for item in pending_items:
                try:
                    # Process based on data type
                    if item.data_type == "observation":
                        await self._sync_observation_data(item)
                    elif item.data_type == "photo":
                        await self._sync_photo_data(item)
                    elif item.data_type == "audio":
                        await self._sync_audio_data(item)
                    
                    # Mark as synced
                    item.sync_status = "synced"
                    item.last_sync_attempt = datetime.utcnow()
                    sync_results["synced"] += 1
                    
                except Exception as e:
                    # Mark as error
                    item.sync_status = "error"
                    item.error_message = str(e)
                    item.sync_attempts += 1
                    item.last_sync_attempt = datetime.utcnow()
                    sync_results["errors"] += 1
                    
                    sync_results["details"].append({
                        "item_id": str(item.id),
                        "error": str(e)
                    })
            
            await self.db.commit()
            
            logger.info("Offline data sync completed", 
                       user_id=user_id, 
                       results=sync_results)
            
            return sync_results
            
        except Exception as e:
            logger.error("Failed to sync offline data", user_id=user_id, error=str(e))
            raise
    
    async def _process_image(self, file_path: Path, content: bytes) -> Dict[str, Any]:
        """Process image and extract metadata"""
        try:
            # Open image
            image = Image.open(file_path)
            
            metadata = {
                "width": image.width,
                "height": image.height,
                "format": image.format
            }
            
            # Extract EXIF data
            if hasattr(image, '_getexif'):
                exif_data = image._getexif()
                if exif_data:
                    exif_dict = {}
                    for tag_id, value in exif_data.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        exif_dict[tag] = str(value)
                    
                    metadata["exif_data"] = exif_dict
                    
                    # Extract GPS coordinates
                    gps_info = exif_data.get(34853)  # GPS tag
                    if gps_info:
                        gps_coords = self._extract_gps_from_exif(gps_info)
                        if gps_coords:
                            metadata.update(gps_coords)
                    
                    # Extract timestamp
                    datetime_original = exif_data.get(36867)  # DateTimeOriginal
                    if datetime_original:
                        try:
                            metadata["taken_at"] = datetime.strptime(datetime_original, "%Y:%m:%d %H:%M:%S")
                        except ValueError:
                            pass
                    
                    # Extract camera info
                    camera_info = {}
                    if 271 in exif_data:  # Make
                        camera_info["make"] = exif_data[271]
                    if 272 in exif_data:  # Model
                        camera_info["model"] = exif_data[272]
                    if camera_info:
                        metadata["camera_info"] = camera_info
            
            return metadata
            
        except Exception as e:
            logger.warning("Failed to process image metadata", error=str(e))
            return {"width": 0, "height": 0}
    
    def _extract_gps_from_exif(self, gps_info: Dict) -> Optional[Dict[str, float]]:
        """Extract GPS coordinates from EXIF data"""
        try:
            def convert_to_degrees(value):
                d, m, s = value
                return d + (m / 60.0) + (s / 3600.0)
            
            lat = gps_info.get(2)  # GPSLatitude
            lat_ref = gps_info.get(1)  # GPSLatitudeRef
            lon = gps_info.get(4)  # GPSLongitude
            lon_ref = gps_info.get(3)  # GPSLongitudeRef
            
            if lat and lon and lat_ref and lon_ref:
                latitude = convert_to_degrees(lat)
                if lat_ref != 'N':
                    latitude = -latitude
                
                longitude = convert_to_degrees(lon)
                if lon_ref != 'E':
                    longitude = -longitude
                
                return {"latitude": latitude, "longitude": longitude}
            
            return None
            
        except Exception:
            return None
    
    async def _process_audio(self, file_path: Path) -> Dict[str, Any]:
        """Process audio file and extract metadata"""
        try:
            # Basic metadata extraction (would use librosa or similar in production)
            file_size = file_path.stat().st_size
            
            # Placeholder for audio processing
            metadata = {
                "duration": 0.0,  # Would extract actual duration
                "sample_rate": 44100,  # Would extract actual sample rate
                "channels": 1  # Would extract actual channel count
            }
            
            return metadata
            
        except Exception as e:
            logger.warning("Failed to process audio metadata", error=str(e))
            return {"duration": 0.0}
    
    async def _generate_thumbnail(self, photo: FieldPhoto):
        """Generate thumbnail for photo"""
        try:
            # Open original image
            image = Image.open(photo.file_path)
            
            # Create thumbnail
            thumbnail_size = (200, 200)
            image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            
            # Save thumbnail
            thumbnail_path = Path(photo.file_path).parent / f"thumb_{photo.filename}"
            image.save(thumbnail_path, "JPEG", quality=85)
            
            # Update database
            photo.thumbnail_path = str(thumbnail_path)
            await self.db.commit()
            
        except Exception as e:
            logger.warning("Failed to generate thumbnail", photo_id=str(photo.id), error=str(e))
    
    async def _calculate_quality_score(self, observation: FieldObservation) -> float:
        """Calculate quality score for observation"""
        try:
            score = 1.0
            
            # GPS accuracy factor
            if observation.accuracy:
                if observation.accuracy <= 5:  # Very accurate
                    score *= 1.0
                elif observation.accuracy <= 10:  # Good
                    score *= 0.9
                elif observation.accuracy <= 20:  # Fair
                    score *= 0.8
                else:  # Poor
                    score *= 0.6
            
            # Description completeness
            if observation.description and len(observation.description) > 50:
                score *= 1.0
            elif observation.description and len(observation.description) > 20:
                score *= 0.9
            else:
                score *= 0.7
            
            # Measurements availability
            if observation.measurements and len(observation.measurements) > 0:
                score *= 1.0
            else:
                score *= 0.8
            
            return min(1.0, score)
            
        except Exception:
            return 0.5  # Default score
    
    async def _queue_audio_transcription(self, audio_note: FieldAudioNote):
        """Queue audio note for transcription"""
        try:
            # Add to Redis queue for background processing
            await self.redis.lpush("audio_transcription_queue", json.dumps({
                "audio_note_id": str(audio_note.id),
                "file_path": audio_note.file_path,
                "queued_at": datetime.utcnow().isoformat()
            }))
            
        except Exception as e:
            logger.warning("Failed to queue audio transcription", error=str(e))
    
    async def _sync_observation_data(self, item: OfflineDataQueue):
        """Sync observation data from offline queue"""
        # Implementation would handle merging offline observation data
        pass
    
    async def _sync_photo_data(self, item: OfflineDataQueue):
        """Sync photo data from offline queue"""
        # Implementation would handle uploading offline photos
        pass
    
    async def _sync_audio_data(self, item: OfflineDataQueue):
        """Sync audio data from offline queue"""
        # Implementation would handle uploading offline audio
        pass

# FastAPI Application
app = FastAPI(
    title="Prometheus Field Data Collection",
    description="Advanced mobile and field data collection system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
redis_client = None
db_engine = None
SessionLocal = None
field_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global redis_client, db_engine, SessionLocal, field_engine
    
    # Initialize Redis
    redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
    
    # Initialize Database
    db_engine = create_async_engine("postgresql+asyncpg://prometheus:prometheus123@localhost/prometheus")
    SessionLocal = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    
    # Initialize field data engine
    async with SessionLocal() as session:
        field_engine = FieldDataCollectionEngine(redis_client, session)
    
    logger.info("Field Data Collection Service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if redis_client:
        await redis_client.close()
    if db_engine:
        await db_engine.dispose()
    
    logger.info("Field Data Collection Service shutdown complete")

# Dependency injection
async def get_db():
    async with SessionLocal() as session:
        yield session

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Extract user from JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, "secret", algorithms=["HS256"])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Health endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    try:
        await redis_client.ping()
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")

# Include API routes
from .api.field_routes import router as field_router
app.include_router(field_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)