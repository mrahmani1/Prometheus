"""
Field Data Collection API Routes
Mobile and field data collection endpoints
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
import json
import base64

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func
from pydantic import BaseModel, Field

from ..main import (
    get_db, get_current_user, FieldSurvey, FieldObservation, FieldPhoto, FieldAudioNote,
    CreateObservationRequest, ObservationResponse, SurveyResponse, GPSCoordinate,
    WeatherConditions, QRCodeData, BarcodeData, field_engine
)
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/field", tags=["field-data"])

# Request/Response Models
class CreateSurveyRequest(BaseModel):
    survey_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    survey_type: str = Field(..., regex="^(geological|geophysical|geochemical|environmental)$")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    location_bounds: Optional[Dict[str, Any]] = None

class UpdateObservationRequest(BaseModel):
    description: Optional[str] = None
    measurements: Optional[List[Dict[str, Any]]] = None
    weather: Optional[WeatherConditions] = None
    equipment_used: Optional[List[str]] = None
    tags: Optional[List[str]] = None

class OfflineDataRequest(BaseModel):
    data_type: str
    data_payload: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SyncStatusResponse(BaseModel):
    total_items: int
    synced: int
    conflicts: int
    errors: int
    details: List[Dict[str, Any]]

class PhotoMetadataResponse(BaseModel):
    id: str
    filename: str
    file_size: int
    width: int
    height: int
    taken_at: Optional[datetime]
    location: Optional[GPSCoordinate]
    camera_info: Optional[Dict[str, Any]]
    thumbnail_url: str

class AudioMetadataResponse(BaseModel):
    id: str
    filename: str
    duration_seconds: float
    file_size: int
    recorded_at: datetime
    transcription: Optional[str]
    transcription_confidence: Optional[float]

class FieldAnalytics(BaseModel):
    total_surveys: int
    active_surveys: int
    total_observations: int
    observations_today: int
    photos_count: int
    audio_notes_count: int
    avg_quality_score: float
    coverage_area_km2: Optional[float]

# Survey Management
@router.post("/surveys", response_model=SurveyResponse)
async def create_survey(
    request: CreateSurveyRequest,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new field survey"""
    try:
        survey = FieldSurvey(
            survey_name=request.survey_name,
            description=request.description,
            created_by=UUID(current_user),
            survey_type=request.survey_type,
            start_date=request.start_date,
            end_date=request.end_date,
            location_bounds=request.location_bounds
        )
        
        db.add(survey)
        await db.commit()
        await db.refresh(survey)
        
        logger.info("Survey created", survey_id=str(survey.id), user_id=current_user)
        
        return SurveyResponse(
            id=str(survey.id),
            survey_name=survey.survey_name,
            description=survey.description,
            created_by=str(survey.created_by),
            created_at=survey.created_at,
            start_date=survey.start_date,
            end_date=survey.end_date,
            survey_type=survey.survey_type,
            status=survey.status,
            observation_count=0,
            location_bounds=survey.location_bounds
        )
        
    except Exception as e:
        logger.error("Failed to create survey", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create survey")

@router.get("/surveys", response_model=List[SurveyResponse])
async def list_surveys(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    status: Optional[str] = Query(None),
    survey_type: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0)
):
    """List user's field surveys"""
    try:
        query = select(FieldSurvey).where(FieldSurvey.created_by == UUID(current_user))
        
        if status:
            query = query.where(FieldSurvey.status == status)
        
        if survey_type:
            query = query.where(FieldSurvey.survey_type == survey_type)
        
        query = query.offset(offset).limit(limit).order_by(desc(FieldSurvey.created_at))
        
        result = await db.execute(query)
        surveys = result.scalars().all()
        
        survey_responses = []
        for survey in surveys:
            # Count observations
            obs_query = select(func.count(FieldObservation.id)).where(
                FieldObservation.survey_id == survey.id
            )
            obs_result = await db.execute(obs_query)
            observation_count = obs_result.scalar() or 0
            
            survey_responses.append(SurveyResponse(
                id=str(survey.id),
                survey_name=survey.survey_name,
                description=survey.description,
                created_by=str(survey.created_by),
                created_at=survey.created_at,
                start_date=survey.start_date,
                end_date=survey.end_date,
                survey_type=survey.survey_type,
                status=survey.status,
                observation_count=observation_count,
                location_bounds=survey.location_bounds
            ))
        
        return survey_responses
        
    except Exception as e:
        logger.error("Failed to list surveys", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list surveys")

@router.get("/surveys/{survey_id}", response_model=SurveyResponse)
async def get_survey(
    survey_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get survey details"""
    try:
        query = select(FieldSurvey).where(FieldSurvey.id == UUID(survey_id))
        result = await db.execute(query)
        survey = result.scalar_one_or_none()
        
        if not survey:
            raise HTTPException(status_code=404, detail="Survey not found")
        
        if str(survey.created_by) != current_user:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Count observations
        obs_query = select(func.count(FieldObservation.id)).where(
            FieldObservation.survey_id == survey.id
        )
        obs_result = await db.execute(obs_query)
        observation_count = obs_result.scalar() or 0
        
        return SurveyResponse(
            id=str(survey.id),
            survey_name=survey.survey_name,
            description=survey.description,
            created_by=str(survey.created_by),
            created_at=survey.created_at,
            start_date=survey.start_date,
            end_date=survey.end_date,
            survey_type=survey.survey_type,
            status=survey.status,
            observation_count=observation_count,
            location_bounds=survey.location_bounds
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get survey", survey_id=survey_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get survey")

# Observation Management
@router.post("/observations", response_model=ObservationResponse)
async def create_observation(
    request: CreateObservationRequest,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new field observation"""
    try:
        # Verify survey exists and user has access
        survey_query = select(FieldSurvey).where(FieldSurvey.id == UUID(request.survey_id))
        survey_result = await db.execute(survey_query)
        survey = survey_result.scalar_one_or_none()
        
        if not survey:
            raise HTTPException(status_code=404, detail="Survey not found")
        
        if str(survey.created_by) != current_user:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create observation using field engine
        observation = await field_engine.create_observation(current_user, request)
        
        return ObservationResponse(
            id=str(observation.id),
            survey_id=str(observation.survey_id),
            observation_type=observation.observation_type,
            description=observation.description,
            timestamp=observation.timestamp,
            location=GPSCoordinate(
                latitude=observation.latitude,
                longitude=observation.longitude,
                elevation=observation.elevation,
                accuracy=observation.accuracy
            ),
            measurements=observation.measurements or [],
            photos=[],  # Will be populated when photos are uploaded
            audio_notes=[],  # Will be populated when audio is uploaded
            weather=WeatherConditions(**observation.weather_conditions) if observation.weather_conditions else None,
            quality_score=observation.quality_score,
            is_validated=observation.is_validated,
            sync_status=observation.sync_status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create observation", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create observation")

@router.get("/observations/{observation_id}", response_model=ObservationResponse)
async def get_observation(
    observation_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get observation details with photos and audio"""
    try:
        query = select(FieldObservation).where(FieldObservation.id == UUID(observation_id))
        result = await db.execute(query)
        observation = result.scalar_one_or_none()
        
        if not observation:
            raise HTTPException(status_code=404, detail="Observation not found")
        
        if str(observation.observer_id) != current_user:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get photos
        photo_query = select(FieldPhoto).where(FieldPhoto.observation_id == observation.id)
        photo_result = await db.execute(photo_query)
        photos = photo_result.scalars().all()
        
        # Get audio notes
        audio_query = select(FieldAudioNote).where(FieldAudioNote.observation_id == observation.id)
        audio_result = await db.execute(audio_query)
        audio_notes = audio_result.scalars().all()
        
        return ObservationResponse(
            id=str(observation.id),
            survey_id=str(observation.survey_id),
            observation_type=observation.observation_type,
            description=observation.description,
            timestamp=observation.timestamp,
            location=GPSCoordinate(
                latitude=observation.latitude,
                longitude=observation.longitude,
                elevation=observation.elevation,
                accuracy=observation.accuracy
            ),
            measurements=observation.measurements or [],
            photos=[
                {
                    "id": str(photo.id),
                    "filename": photo.filename,
                    "taken_at": photo.taken_at.isoformat() if photo.taken_at else None,
                    "thumbnail_url": f"/api/v1/field/photos/{photo.id}/thumbnail"
                }
                for photo in photos
            ],
            audio_notes=[
                {
                    "id": str(audio.id),
                    "filename": audio.filename,
                    "duration": audio.duration_seconds,
                    "recorded_at": audio.recorded_at.isoformat()
                }
                for audio in audio_notes
            ],
            weather=WeatherConditions(**observation.weather_conditions) if observation.weather_conditions else None,
            quality_score=observation.quality_score,
            is_validated=observation.is_validated,
            sync_status=observation.sync_status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get observation", observation_id=observation_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get observation")

@router.get("/surveys/{survey_id}/observations", response_model=List[ObservationResponse])
async def list_survey_observations(
    survey_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    observation_type: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0)
):
    """List observations for a survey"""
    try:
        # Verify survey access
        survey_query = select(FieldSurvey).where(FieldSurvey.id == UUID(survey_id))
        survey_result = await db.execute(survey_query)
        survey = survey_result.scalar_one_or_none()
        
        if not survey:
            raise HTTPException(status_code=404, detail="Survey not found")
        
        if str(survey.created_by) != current_user:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get observations
        query = select(FieldObservation).where(FieldObservation.survey_id == UUID(survey_id))
        
        if observation_type:
            query = query.where(FieldObservation.observation_type == observation_type)
        
        query = query.offset(offset).limit(limit).order_by(desc(FieldObservation.timestamp))
        
        result = await db.execute(query)
        observations = result.scalars().all()
        
        observation_responses = []
        for obs in observations:
            observation_responses.append(ObservationResponse(
                id=str(obs.id),
                survey_id=str(obs.survey_id),
                observation_type=obs.observation_type,
                description=obs.description,
                timestamp=obs.timestamp,
                location=GPSCoordinate(
                    latitude=obs.latitude,
                    longitude=obs.longitude,
                    elevation=obs.elevation,
                    accuracy=obs.accuracy
                ),
                measurements=obs.measurements or [],
                photos=[],  # Simplified for list view
                audio_notes=[],  # Simplified for list view
                weather=WeatherConditions(**obs.weather_conditions) if obs.weather_conditions else None,
                quality_score=obs.quality_score,
                is_validated=obs.is_validated,
                sync_status=obs.sync_status
            ))
        
        return observation_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list observations", survey_id=survey_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list observations")

# Photo Management
@router.post("/observations/{observation_id}/photos", response_model=PhotoMetadataResponse)
async def upload_photo(
    observation_id: str,
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload photo for observation"""
    try:
        # Verify observation exists and user has access
        obs_query = select(FieldObservation).where(FieldObservation.id == UUID(observation_id))
        obs_result = await db.execute(obs_query)
        observation = obs_result.scalar_one_or_none()
        
        if not observation:
            raise HTTPException(status_code=404, detail="Observation not found")
        
        if str(observation.observer_id) != current_user:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Process photo using field engine
        photo = await field_engine.process_photo_upload(observation_id, file)
        
        return PhotoMetadataResponse(
            id=str(photo.id),
            filename=photo.filename,
            file_size=photo.file_size,
            width=photo.width or 0,
            height=photo.height or 0,
            taken_at=photo.taken_at,
            location=GPSCoordinate(
                latitude=photo.latitude,
                longitude=photo.longitude
            ) if photo.latitude and photo.longitude else None,
            camera_info=photo.camera_info,
            thumbnail_url=f"/api/v1/field/photos/{photo.id}/thumbnail"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload photo", observation_id=observation_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to upload photo")

@router.get("/photos/{photo_id}")
async def get_photo(
    photo_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get full resolution photo"""
    try:
        query = select(FieldPhoto).where(FieldPhoto.id == UUID(photo_id))
        result = await db.execute(query)
        photo = result.scalar_one_or_none()
        
        if not photo:
            raise HTTPException(status_code=404, detail="Photo not found")
        
        # Verify access through observation
        obs_query = select(FieldObservation).where(FieldObservation.id == photo.observation_id)
        obs_result = await db.execute(obs_query)
        observation = obs_result.scalar_one_or_none()
        
        if not observation or str(observation.observer_id) != current_user:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(
            photo.file_path,
            media_type=photo.mime_type,
            filename=photo.original_filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get photo", photo_id=photo_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get photo")

@router.get("/photos/{photo_id}/thumbnail")
async def get_photo_thumbnail(
    photo_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get photo thumbnail"""
    try:
        query = select(FieldPhoto).where(FieldPhoto.id == UUID(photo_id))
        result = await db.execute(query)
        photo = result.scalar_one_or_none()
        
        if not photo:
            raise HTTPException(status_code=404, detail="Photo not found")
        
        # Verify access through observation
        obs_query = select(FieldObservation).where(FieldObservation.id == photo.observation_id)
        obs_result = await db.execute(obs_query)
        observation = obs_result.scalar_one_or_none()
        
        if not observation or str(observation.observer_id) != current_user:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not photo.thumbnail_path:
            raise HTTPException(status_code=404, detail="Thumbnail not available")
        
        return FileResponse(
            photo.thumbnail_path,
            media_type="image/jpeg"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get thumbnail", photo_id=photo_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get thumbnail")

# Audio Management
@router.post("/observations/{observation_id}/audio", response_model=AudioMetadataResponse)
async def upload_audio(
    observation_id: str,
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload audio note for observation"""
    try:
        # Verify observation exists and user has access
        obs_query = select(FieldObservation).where(FieldObservation.id == UUID(observation_id))
        obs_result = await db.execute(obs_query)
        observation = obs_result.scalar_one_or_none()
        
        if not observation:
            raise HTTPException(status_code=404, detail="Observation not found")
        
        if str(observation.observer_id) != current_user:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Process audio using field engine
        audio_note = await field_engine.process_audio_upload(observation_id, file)
        
        return AudioMetadataResponse(
            id=str(audio_note.id),
            filename=audio_note.filename,
            duration_seconds=audio_note.duration_seconds or 0.0,
            file_size=audio_note.file_size,
            recorded_at=audio_note.recorded_at,
            transcription=audio_note.transcription,
            transcription_confidence=audio_note.transcription_confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload audio", observation_id=observation_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to upload audio")

# QR Code and Barcode Scanning
@router.post("/scan/qr", response_model=List[QRCodeData])
async def scan_qr_code(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    """Scan QR codes from uploaded image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Scan QR codes using field engine
        qr_codes = await field_engine.scan_qr_code(image_data)
        
        return qr_codes
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to scan QR code", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to scan QR code")

@router.post("/generate/qr")
async def generate_qr_code(
    data: QRCodeData,
    current_user: str = Depends(get_current_user)
):
    """Generate QR code for data"""
    try:
        # Generate QR code using field engine
        qr_image_data = await field_engine.generate_qr_code(data)
        
        # Return as streaming response
        import io
        return StreamingResponse(
            io.BytesIO(qr_image_data),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=qr_code_{data.id}.png"}
        )
        
    except Exception as e:
        logger.error("Failed to generate QR code", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate QR code")

# Offline Data Management
@router.post("/offline/queue", response_model=Dict[str, str])
async def queue_offline_data(
    request: OfflineDataRequest,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Queue data for offline synchronization"""
    try:
        from ..main import OfflineDataQueue
        
        offline_item = OfflineDataQueue(
            user_id=UUID(current_user),
            data_type=request.data_type,
            data_payload=request.data_payload,
            created_at=request.created_at
        )
        
        db.add(offline_item)
        await db.commit()
        await db.refresh(offline_item)
        
        logger.info("Offline data queued", item_id=str(offline_item.id), user_id=current_user)
        
        return {"message": "Data queued for synchronization", "item_id": str(offline_item.id)}
        
    except Exception as e:
        logger.error("Failed to queue offline data", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to queue offline data")

@router.post("/offline/sync", response_model=SyncStatusResponse)
async def sync_offline_data(
    current_user: str = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Synchronize offline data queue"""
    try:
        # Sync offline data using field engine
        sync_results = await field_engine.sync_offline_data(current_user)
        
        return SyncStatusResponse(**sync_results)
        
    except Exception as e:
        logger.error("Failed to sync offline data", user_id=current_user, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to sync offline data")

# Analytics and Reporting
@router.get("/analytics", response_model=FieldAnalytics)
async def get_field_analytics(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    days: int = Query(30, ge=1, le=365)
):
    """Get field data collection analytics"""
    try:
        since_date = datetime.utcnow() - timedelta(days=days)
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Count surveys
        survey_query = select(func.count(FieldSurvey.id)).where(
            and_(
                FieldSurvey.created_by == UUID(current_user),
                FieldSurvey.created_at >= since_date
            )
        )
        survey_result = await db.execute(survey_query)
        total_surveys = survey_result.scalar() or 0
        
        # Count active surveys
        active_survey_query = select(func.count(FieldSurvey.id)).where(
            and_(
                FieldSurvey.created_by == UUID(current_user),
                FieldSurvey.status == "active"
            )
        )
        active_survey_result = await db.execute(active_survey_query)
        active_surveys = active_survey_result.scalar() or 0
        
        # Count observations
        obs_query = select(func.count(FieldObservation.id)).where(
            and_(
                FieldObservation.observer_id == UUID(current_user),
                FieldObservation.timestamp >= since_date
            )
        )
        obs_result = await db.execute(obs_query)
        total_observations = obs_result.scalar() or 0
        
        # Count today's observations
        today_obs_query = select(func.count(FieldObservation.id)).where(
            and_(
                FieldObservation.observer_id == UUID(current_user),
                FieldObservation.timestamp >= today_start
            )
        )
        today_obs_result = await db.execute(today_obs_query)
        observations_today = today_obs_result.scalar() or 0
        
        # Count photos
        photo_query = select(func.count(FieldPhoto.id)).where(
            FieldPhoto.observation_id.in_(
                select(FieldObservation.id).where(
                    FieldObservation.observer_id == UUID(current_user)
                )
            )
        )
        photo_result = await db.execute(photo_query)
        photos_count = photo_result.scalar() or 0
        
        # Count audio notes
        audio_query = select(func.count(FieldAudioNote.id)).where(
            FieldAudioNote.observation_id.in_(
                select(FieldObservation.id).where(
                    FieldObservation.observer_id == UUID(current_user)
                )
            )
        )
        audio_result = await db.execute(audio_query)
        audio_notes_count = audio_result.scalar() or 0
        
        # Calculate average quality score
        quality_query = select(func.avg(FieldObservation.quality_score)).where(
            and_(
                FieldObservation.observer_id == UUID(current_user),
                FieldObservation.timestamp >= since_date
            )
        )
        quality_result = await db.execute(quality_query)
        avg_quality_score = float(quality_result.scalar() or 0.0)
        
        return FieldAnalytics(
            total_surveys=total_surveys,
            active_surveys=active_surveys,
            total_observations=total_observations,
            observations_today=observations_today,
            photos_count=photos_count,
            audio_notes_count=audio_notes_count,
            avg_quality_score=round(avg_quality_score, 2),
            coverage_area_km2=None  # Would calculate from observation locations
        )
        
    except Exception as e:
        logger.error("Failed to get analytics", user_id=current_user, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get analytics")