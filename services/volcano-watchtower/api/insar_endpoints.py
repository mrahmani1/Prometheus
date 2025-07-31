"""
InSAR Analysis API Endpoints

This module provides REST API endpoints for InSAR deformation analysis,
including data upload, processing, and results retrieval.

Requirements addressed: 3.3, 1.2
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import logging
import asyncio
import json
import io

from ..core.insar_analysis import (
    InSARDeformationAnalyzer, InSARData, DeformationResult,
    load_insar_data, create_synthetic_insar_time_series
)
from ..core.insar_integration import InSARMonitoringService

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/insar", tags=["InSAR Analysis"])

# Global instances (in production, these would be dependency-injected)
insar_analyzer = InSARDeformationAnalyzer()
monitoring_service = InSARMonitoringService(insar_analyzer)


# Pydantic models for API requests/responses
class InSARProcessingRequest(BaseModel):
    """Request model for InSAR processing"""
    volcano_id: str = Field(..., description="Volcano identifier")
    acquisition_date_1: datetime = Field(..., description="First acquisition date")
    acquisition_date_2: datetime = Field(..., description="Second acquisition date")
    wavelength: float = Field(0.056, description="Radar wavelength in meters")
    incidence_angle: float = Field(23.0, description="Incidence angle in degrees")
    heading_angle: float = Field(-167.0, description="Satellite heading angle")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DeformationPattern(BaseModel):
    """Model for deformation pattern results"""
    pattern_type: str
    confidence: float
    location: Optional[List[float]] = None
    magnitude_m: Optional[float] = None
    spatial_extent: Optional[Dict[str, Any]] = None


class ProcessingResult(BaseModel):
    """Model for processing results"""
    volcano_id: str
    processing_time: datetime
    max_displacement_m: float
    mean_uncertainty_m: float
    quality_score: float
    patterns_detected: int
    alerts_generated: int
    alert_levels: List[int]


class QualityMetrics(BaseModel):
    """Model for quality metrics"""
    mean_coherence: float
    coherence_std: float
    high_coherence_fraction: float
    mean_uncertainty_m: float
    max_uncertainty_m: float
    displacement_snr: float
    spatial_coverage: float
    overall_quality_score: float


class InSARAnalysisResponse(BaseModel):
    """Response model for InSAR analysis"""
    processing_result: ProcessingResult
    quality_metrics: QualityMetrics
    deformation_patterns: List[DeformationPattern]
    alerts: List[Dict[str, Any]]
    displacement_statistics: Dict[str, float]


class TimeSeriesRequest(BaseModel):
    """Request model for time series analysis"""
    volcano_id: str
    max_acquisitions: int = Field(50, description="Maximum acquisitions to use")


class MonitoringStatusResponse(BaseModel):
    """Response model for monitoring status"""
    volcano_id: str
    last_update: Optional[datetime]
    total_acquisitions: int
    current_max_displacement_m: float
    current_quality_score: float
    active_patterns: int
    monitoring_duration_days: Optional[int]
    current_max_velocity_m_per_year: Optional[float] = None


@router.post("/process", response_model=InSARAnalysisResponse)
async def process_insar_data(
    request: InSARProcessingRequest,
    background_tasks: BackgroundTasks
) -> InSARAnalysisResponse:
    """
    Process InSAR interferogram data and return deformation analysis results.
    
    This endpoint accepts InSAR processing parameters and returns comprehensive
    deformation analysis including atmospheric correction, noise reduction,
    pattern recognition, and uncertainty estimation.
    """
    try:
        logger.info(f"Processing InSAR data for volcano {request.volcano_id}")
        
        # For demonstration, create synthetic InSAR data
        # In production, this would load actual InSAR files
        insar_data = InSARData(
            interferogram=np.random.randn(256, 256) * 0.1,
            coherence=np.random.uniform(0.3, 0.9, (256, 256)),
            amplitude=np.random.uniform(1000, 5000, (256, 256)),
            coordinates=np.zeros((256, 256, 2)),
            acquisition_dates=(request.acquisition_date_1, request.acquisition_date_2),
            wavelength=request.wavelength,
            incidence_angle=np.full((256, 256), request.incidence_angle),
            heading_angle=request.heading_angle,
            metadata=request.metadata
        )
        
        # Add synthetic deformation for demonstration
        y, x = np.ogrid[:256, :256]
        center_y, center_x = 128, 128
        r = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        synthetic_deformation = 0.03 * np.exp(-r**2 / (2 * 30**2))
        insar_data.interferogram += synthetic_deformation * 4 * np.pi / request.wavelength
        
        # Process through monitoring service
        result = await monitoring_service.process_new_insar_data(insar_data, request.volcano_id)
        
        # Convert results to response format
        deformation_result = result['deformation_result']
        processing_summary = result['processing_summary']
        alerts = result['alerts']
        
        # Calculate displacement statistics
        displacement_stats = {
            'min_displacement_m': float(np.nanmin(deformation_result.displacement_map)),
            'max_displacement_m': float(np.nanmax(deformation_result.displacement_map)),
            'mean_displacement_m': float(np.nanmean(deformation_result.displacement_map)),
            'std_displacement_m': float(np.nanstd(deformation_result.displacement_map)),
            'rms_displacement_m': float(np.sqrt(np.nanmean(deformation_result.displacement_map**2)))
        }
        
        # Convert deformation patterns
        patterns = [
            DeformationPattern(
                pattern_type=pattern.get('pattern_type', 'unknown'),
                confidence=pattern.get('confidence', 0.0),
                location=pattern.get('location'),
                magnitude_m=pattern.get('magnitude_m'),
                spatial_extent=pattern.get('spatial_extent')
            )
            for pattern in deformation_result.deformation_patterns
        ]
        
        response = InSARAnalysisResponse(
            processing_result=ProcessingResult(**processing_summary),
            quality_metrics=QualityMetrics(**deformation_result.quality_metrics),
            deformation_patterns=patterns,
            alerts=alerts,
            displacement_statistics=displacement_stats
        )
        
        logger.info(f"InSAR processing completed for {request.volcano_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing InSAR data: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/upload")
async def upload_insar_file(
    volcano_id: str,
    file: UploadFile = File(...),
    acquisition_date_1: datetime = None,
    acquisition_date_2: datetime = None,
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Upload InSAR data file for processing.
    
    Accepts various InSAR data formats and queues them for processing.
    """
    try:
        logger.info(f"Uploading InSAR file for volcano {volcano_id}: {file.filename}")
        
        # Validate file type
        allowed_extensions = ['.tif', '.tiff', '.h5', '.nc', '.bin']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {allowed_extensions}"
            )
        
        # Read file content
        content = await file.read()
        
        # In production, this would parse the actual file format
        # For now, we'll simulate processing
        file_info = {
            'filename': file.filename,
            'size_bytes': len(content),
            'content_type': file.content_type,
            'upload_time': datetime.now()
        }
        
        # Queue for background processing
        if background_tasks:
            background_tasks.add_task(
                _process_uploaded_file,
                volcano_id,
                content,
                file_info,
                acquisition_date_1,
                acquisition_date_2
            )
        
        return {
            'status': 'uploaded',
            'volcano_id': volcano_id,
            'file_info': file_info,
            'message': 'File uploaded successfully and queued for processing'
        }
        
    except Exception as e:
        logger.error(f"Error uploading InSAR file: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/time-series", response_model=Dict[str, Any])
async def process_time_series(request: TimeSeriesRequest) -> Dict[str, Any]:
    """
    Process accumulated time series data for velocity estimation.
    
    Analyzes the time series of InSAR acquisitions to estimate
    deformation velocity and detect temporal trends.
    """
    try:
        logger.info(f"Processing time series for volcano {request.volcano_id}")
        
        result = await monitoring_service.process_time_series_update(
            request.volcano_id, 
            request.max_acquisitions
        )
        
        if result.get('status') == 'insufficient_data':
            raise HTTPException(
                status_code=400,
                detail="Insufficient time series data for analysis"
            )
        
        # Convert numpy arrays to lists for JSON serialization
        velocity_result = result.get('velocity_result')
        if velocity_result and velocity_result.velocity_map is not None:
            velocity_stats = {
                'min_velocity_m_per_year': float(np.nanmin(velocity_result.velocity_map)),
                'max_velocity_m_per_year': float(np.nanmax(velocity_result.velocity_map)),
                'mean_velocity_m_per_year': float(np.nanmean(velocity_result.velocity_map)),
                'std_velocity_m_per_year': float(np.nanstd(velocity_result.velocity_map))
            }
            result['velocity_statistics'] = velocity_stats
        
        logger.info(f"Time series processing completed for {request.volcano_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing time series: {e}")
        raise HTTPException(status_code=500, detail=f"Time series processing failed: {str(e)}")


@router.get("/status/{volcano_id}", response_model=MonitoringStatusResponse)
async def get_monitoring_status(volcano_id: str) -> MonitoringStatusResponse:
    """
    Get current monitoring status for a volcano.
    
    Returns comprehensive monitoring statistics including data availability,
    current deformation state, and quality metrics.
    """
    try:
        status = monitoring_service.get_monitoring_status(volcano_id)
        
        if status.get('status') == 'no_data':
            raise HTTPException(
                status_code=404,
                detail=f"No monitoring data available for volcano {volcano_id}"
            )
        
        return MonitoringStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


@router.get("/patterns/{volcano_id}")
async def get_deformation_patterns(volcano_id: str, limit: int = 10) -> Dict[str, Any]:
    """
    Get recent deformation patterns for a volcano.
    
    Returns the most recent deformation patterns detected
    for the specified volcano.
    """
    try:
        # Get recent patterns from monitoring service
        if not monitoring_service.deformation_history:
            raise HTTPException(
                status_code=404,
                detail=f"No deformation data available for volcano {volcano_id}"
            )
        
        recent_results = monitoring_service.deformation_history[-limit:]
        
        all_patterns = []
        for i, result in enumerate(recent_results):
            for pattern in result.deformation_patterns:
                pattern_info = {
                    'result_index': len(monitoring_service.deformation_history) - limit + i,
                    'processing_date': result.processing_metadata.get('processing_date'),
                    **pattern
                }
                all_patterns.append(pattern_info)
        
        return {
            'volcano_id': volcano_id,
            'total_patterns': len(all_patterns),
            'patterns': all_patterns,
            'query_limit': limit
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting deformation patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern retrieval failed: {str(e)}")


@router.post("/synthetic-demo")
async def create_synthetic_demo(
    volcano_id: str = "demo_volcano",
    num_acquisitions: int = 5
) -> Dict[str, Any]:
    """
    Create synthetic InSAR data for demonstration purposes.
    
    Generates a time series of synthetic InSAR acquisitions
    and processes them to demonstrate the analysis capabilities.
    """
    try:
        logger.info(f"Creating synthetic demo for {volcano_id} with {num_acquisitions} acquisitions")
        
        # Create synthetic time series
        time_series = create_synthetic_insar_time_series(num_acquisitions)
        
        # Process each acquisition
        results = []
        for i, insar_data in enumerate(time_series):
            result = await monitoring_service.process_new_insar_data(insar_data, volcano_id)
            
            # Simplify result for JSON response
            simplified_result = {
                'acquisition_number': i,
                'processing_summary': result['processing_summary'],
                'alerts_count': len(result['alerts']),
                'patterns_count': len(result['deformation_result'].deformation_patterns)
            }
            results.append(simplified_result)
        
        # Process time series
        time_series_result = await monitoring_service.process_time_series_update(volcano_id)
        
        # Get final status
        status = monitoring_service.get_monitoring_status(volcano_id)
        
        return {
            'demo_info': {
                'volcano_id': volcano_id,
                'acquisitions_processed': num_acquisitions,
                'demo_created': datetime.now()
            },
            'individual_results': results,
            'time_series_analysis': time_series_result,
            'final_status': status
        }
        
    except Exception as e:
        logger.error(f"Error creating synthetic demo: {e}")
        raise HTTPException(status_code=500, detail=f"Demo creation failed: {str(e)}")


@router.delete("/data/{volcano_id}")
async def clear_volcano_data(volcano_id: str) -> Dict[str, str]:
    """
    Clear all stored data for a volcano.
    
    Removes all historical InSAR data and analysis results
    for the specified volcano.
    """
    try:
        # Clear monitoring service data
        # In production, this would be more sophisticated with proper data management
        monitoring_service.deformation_history.clear()
        monitoring_service.time_series_data.clear()
        
        logger.info(f"Cleared all data for volcano {volcano_id}")
        
        return {
            'status': 'success',
            'message': f'All data cleared for volcano {volcano_id}'
        }
        
    except Exception as e:
        logger.error(f"Error clearing volcano data: {e}")
        raise HTTPException(status_code=500, detail=f"Data clearing failed: {str(e)}")


# Background task functions
async def _process_uploaded_file(
    volcano_id: str,
    file_content: bytes,
    file_info: Dict[str, Any],
    acquisition_date_1: Optional[datetime],
    acquisition_date_2: Optional[datetime]
):
    """Background task to process uploaded InSAR file"""
    try:
        logger.info(f"Background processing of {file_info['filename']} for {volcano_id}")
        
        # In production, this would parse the actual file format
        # For now, simulate processing with synthetic data
        metadata = {
            'filename': file_info['filename'],
            'upload_time': file_info['upload_time'],
            'file_size': file_info['size_bytes']
        }
        
        # Create synthetic data based on file
        insar_data = load_insar_data(file_info['filename'], metadata)
        
        if acquisition_date_1 and acquisition_date_2:
            insar_data.acquisition_dates = (acquisition_date_1, acquisition_date_2)
        
        # Process the data
        result = await monitoring_service.process_new_insar_data(insar_data, volcano_id)
        
        logger.info(f"Background processing completed for {file_info['filename']}")
        
    except Exception as e:
        logger.error(f"Background processing failed for {file_info['filename']}: {e}")


# Health check endpoint
@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for InSAR analysis service"""
    try:
        # Test analyzer functionality
        test_
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
            resample_config = config['resample']
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
        
        return {'method': 'none', 'anomaly_count': 0}
randn(64, 64)
        _ = insar_analyzer._reduce_noise(test_data)
        
        return {
            'status': 'healthy',
            'service': 'InSAR Analysis',
            'analyzer_device': insar_analyzer.device,
            'timestamp': datetime.now(),
            'models_loaded': {
                'atmospheric_model': insar_analyzer.atmospheric_model is not None,
                'pattern_model': insar_analyzer.pattern_model is not None,
                'noise_model': insar_analyzer.noise_model is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now()
        }