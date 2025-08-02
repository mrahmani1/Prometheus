"""
API Routes for Volcano Watchtower Service
RESTful endpoints for volcanic risk assessment and monitoring
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import structlog
from core.alert_system import AlertLevelManager, NotificationSystem
from core.eruption_forecasting import ProbabilisticEruptionForecaster
from core.magma_reservoir import MagmaReservoirEstimator
from core.risk_assessment import VolcanicRiskAssessmentEngine
from fastapi import APIRouter, BackgroundTasks, HTTPException, Path, Query
from models.volcanic_data import AlertLevel
from pydantic import BaseModel, Field

from .insar_endpoints import router as insar_router

logger = structlog.get_logger()

# Pydantic models for API requests/responses


class RiskAssessmentRequest(BaseModel):
    """Request model for risk assessment"""

    volcano_id: str = Field(..., description="Unique volcano identifier")
    include_forecast: bool = Field(True, description="Include eruption forecast")
    include_reservoir: bool = Field(True, description="Include reservoir parameters")
    timeframe_hours: int = Field(24, description="Assessment timeframe in hours")


class RiskAssessmentResponse(BaseModel):
    """Response model for risk assessment"""

    volcano_id: str
    timestamp: str
    alert_level: int
    alert_description: str
    composite_risk_score: float
    eruption_probability: Dict[str, float]
    contributing_factors: List[str]
    confidence: float
    hazard_zones: Dict[str, List[float]]
    recommended_actions: List[str]


class AlertUpdateRequest(BaseModel):
    """Request model for alert level updates"""

    volcano_id: str
    alert_level: int = Field(..., ge=0, le=5)
    description: str
    recommended_actions: List[str]
    validity_hours: int = Field(6, description="Alert validity in hours")
    confidence: float = Field(..., ge=0.0, le=1.0)


class ForecastRequest(BaseModel):
    """Request model for eruption forecasting"""

    volcano_id: str
    timeframes: List[str] = Field(
        default=["1_hour", "6_hours", "24_hours", "1_week", "1_month"],
        description="Forecast timeframes",
    )
    include_uncertainty: bool = Field(True, description="Include uncertainty estimates")


class NotificationRequest(BaseModel):
    """Request model for sending notifications"""

    volcano_id: str
    message: str
    priority: str = Field("medium", description="Notification priority")
    channels: List[str] = Field(
        default=["email", "sms"], description="Notification channels"
    )
    targets: List[str] = Field(
        default=["observatory_sta"], description="Notification targets"
    )


class VolcanoStatusResponse(BaseModel):
    """Response model for volcano status"""

    volcano_id: str
    current_alert_level: int
    last_assessment: str
    monitoring_status: str
    data_quality: float
    active_alerts: int
    last_eruption: Optional[str] = None


def create_routes(
    app,
    risk_engine: VolcanicRiskAssessmentEngine,
    forecaster: ProbabilisticEruptionForecaster,
    reservoir_estimator: MagmaReservoirEstimator,
    alert_manager: AlertLevelManager,
    notification_system: NotificationSystem,
):
    """Create and register API routes"""

    router = APIRouter(prefix="/api/v1", tags=["volcano-watchtower"])

    @router.get("/status")
    async def get_service_status():
        """Get service status and health information"""
        return {
            "service": "volcano-watchtower",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "components": {
                "risk_engine": risk_engine is not None,
                "forecaster": forecaster is not None,
                "reservoir_estimator": reservoir_estimator is not None,
                "alert_manager": alert_manager is not None,
                "notification_system": notification_system is not None,
            },
        }

    @router.get("/volcanoes")
    async def list_monitored_volcanoes():
        """Get list of monitored volcanoes"""
        # This would query the monitoring database
        # For now, return a static list
        volcanoes = [
            {
                "volcano_id": "yellowstone",
                "name": "Yellowstone Caldera",
                "location": {
                    "latitude": 44.428,
                    "longitude": -110.588,
                    "elevation": 2400,
                },
                "type": "caldera",
                "monitoring_status": "active",
            },
            {
                "volcano_id": "kilauea",
                "name": "Kilauea",
                "location": {
                    "latitude": 19.421,
                    "longitude": -155.287,
                    "elevation": 1222,
                },
                "type": "shield",
                "monitoring_status": "active",
            },
            {
                "volcano_id": "etna",
                "name": "Mount Etna",
                "location": {
                    "latitude": 37.748,
                    "longitude": 14.999,
                    "elevation": 3329,
                },
                "type": "stratovolcano",
                "monitoring_status": "active",
            },
            {
                "volcano_id": "stromboli",
                "name": "Stromboli",
                "location": {"latitude": 38.789, "longitude": 15.213, "elevation": 924},
                "type": "stratovolcano",
                "monitoring_status": "active",
            },
            {
                "volcano_id": "sakurajima",
                "name": "Sakurajima",
                "location": {
                    "latitude": 31.585,
                    "longitude": 130.657,
                    "elevation": 1117,
                },
                "type": "stratovolcano",
                "monitoring_status": "active",
            },
        ]

        return {"volcanoes": volcanoes, "total": len(volcanoes)}

    @router.get("/volcanoes/{volcano_id}/status", response_model=VolcanoStatusResponse)
    async def get_volcano_status(
        volcano_id: str = Path(..., description="Volcano identifier")
    ):
        """Get current status for a specific volcano"""
        try:
            # Get current alert level
            current_alert = await alert_manager.get_current_alert_level(volcano_id)

            # Get alert statistics
            alert_stats = await alert_manager.get_alert_statistics(volcano_id, days=30)

            return VolcanoStatusResponse(
                volcano_id=volcano_id,
                current_alert_level=current_alert.level,
                last_assessment=datetime.utcnow().isoformat(),
                monitoring_status="active",
                data_quality=0.85,  # Would be calculated from actual data
                active_alerts=1 if current_alert.level > 0 else 0,
                last_eruption=None,  # Would be queried from eruption database
            )

        except Exception as e:
            logger.error(
                "Error getting volcano status", volcano_id=volcano_id, error=str(e)
            )
            raise HTTPException(
                status_code=500, detail=f"Error getting volcano status: {str(e)}"
            )

    @router.post("/assess", response_model=RiskAssessmentResponse)
    async def assess_volcanic_risk(request: RiskAssessmentRequest):
        """Perform comprehensive volcanic risk assessment"""
        try:
            logger.info("Risk assessment requested", volcano_id=request.volcano_id)

            # Perform risk assessment
            risk_assessment = await risk_engine.assess_volcanic_risk(request.volcano_id)

            # Convert hazard zones to serializable format
            hazard_zones_serializable = {}
            for hazard_type, zones in risk_assessment.hazard_zones.items():
                hazard_zones_serializable[hazard_type] = (
                    zones.tolist() if hasattr(zones, "tolist") else zones
                )

            response = RiskAssessmentResponse(
                volcano_id=risk_assessment.volcano_id,
                timestamp=risk_assessment.timestamp.isoformat(),
                alert_level=risk_assessment.alert_level.level,
                alert_description=risk_assessment.alert_level.description,
                composite_risk_score=risk_assessment.composite_risk_score,
                eruption_probability=risk_assessment.eruption_probability,
                contributing_factors=risk_assessment.contributing_factors,
                confidence=risk_assessment.model_ensemble_agreement,
                hazard_zones=hazard_zones_serializable,
                recommended_actions=risk_assessment.alert_level.recommended_actions,
            )

            return response

        except Exception as e:
            logger.error(
                "Error in risk assessment", volcano_id=request.volcano_id, error=str(e)
            )
            raise HTTPException(
                status_code=500, detail=f"Risk assessment failed: {str(e)}"
            )

    @router.post("/forecast")
    async def generate_eruption_forecast(request: ForecastRequest):
        """Generate probabilistic eruption forecast"""
        try:
            logger.info("Eruption forecast requested", volcano_id=request.volcano_id)

            # This would gather real monitoring data
            # For now, use empty monitoring data
            monitoring_data = {
                "seismic": [],
                "deformation": [],
                "geochemical": [],
                "thermal": [],
            }

            # Generate forecast
            forecast = await forecaster.generate_probabilistic_forecast(
                request.volcano_id, monitoring_data
            )

            response = {
                "volcano_id": forecast.volcano_id,
                "timestamp": forecast.timestamp.isoformat(),
                "probability_by_timeframe": forecast.probability_by_timeframe,
                "uncertainty": forecast.uncertainty,
                "confidence": forecast.confidence,
                "key_indicators": forecast.key_indicators,
                "model_ensemble_results": (
                    forecast.model_ensemble_results
                    if request.include_uncertainty
                    else None
                ),
            }

            return response

        except Exception as e:
            logger.error(
                "Error generating forecast", volcano_id=request.volcano_id, error=str(e)
            )
            raise HTTPException(
                status_code=500, detail=f"Forecast generation failed: {str(e)}"
            )

    @router.get("/volcanoes/{volcano_id}/alerts")
    async def get_volcano_alerts(
        volcano_id: str = Path(..., description="Volcano identifier"),
        days: int = Query(30, description="Number of days to look back"),
    ):
        """Get alert history for a volcano"""
        try:
            # Get alert statistics
            alert_stats = await alert_manager.get_alert_statistics(volcano_id, days)

            # Get current alert
            current_alert = await alert_manager.get_current_alert_level(volcano_id)

            return {
                "volcano_id": volcano_id,
                "current_alert": {
                    "level": current_alert.level,
                    "description": current_alert.description,
                    "recommended_actions": current_alert.recommended_actions,
                    "confidence": current_alert.confidence,
                },
                "statistics": alert_stats,
                "period_days": days,
            }

        except Exception as e:
            logger.error("Error getting alerts", volcano_id=volcano_id, error=str(e))
            raise HTTPException(
                status_code=500, detail=f"Error getting alerts: {str(e)}"
            )

    @router.post("/alerts/update")
    async def update_alert_level(
        request: AlertUpdateRequest, background_tasks: BackgroundTasks
    ):
        """Update alert level for a volcano"""
        try:
            logger.info(
                "Alert level update requested",
                volcano_id=request.volcano_id,
                level=request.alert_level,
            )

            # Create new alert level
            new_alert = AlertLevel(
                level=request.alert_level,
                description=request.description,
                recommended_actions=request.recommended_actions,
                validity_period=(
                    datetime.utcnow(),
                    datetime.utcnow() + timedelta(hours=request.validity_hours),
                ),
                confidence=request.confidence,
            )

            # Update alert level
            changed = await alert_manager.update_alert_level(
                request.volcano_id, new_alert
            )

            if changed:
                # Send notifications in background
                background_tasks.add_task(
                    _send_alert_notification,
                    request.volcano_id,
                    new_alert,
                    notification_system,
                )

            return {
                "volcano_id": request.volcano_id,
                "alert_level": request.alert_level,
                "description": request.description,
                "updated": changed,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(
                "Error updating alert", volcano_id=request.volcano_id, error=str(e)
            )
            raise HTTPException(
                status_code=500, detail=f"Alert update failed: {str(e)}"
            )

    @router.post("/notifications/send")
    async def send_notification(
        request: NotificationRequest, background_tasks: BackgroundTasks
    ):
        """Send notification for a volcano"""
        try:
            logger.info("Notification requested", volcano_id=request.volcano_id)

            # Send notification in background
            background_tasks.add_task(
                _send_custom_notification,
                request.volcano_id,
                request.message,
                request.priority,
                request.channels,
                request.targets,
                notification_system,
            )

            return {
                "volcano_id": request.volcano_id,
                "message": "Notification queued for delivery",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(
                "Error sending notification",
                volcano_id=request.volcano_id,
                error=str(e),
            )
            raise HTTPException(
                status_code=500, detail=f"Notification failed: {str(e)}"
            )

    @router.get("/notifications/statistics")
    async def get_notification_statistics(
        days: int = Query(30, description="Number of days to look back")
    ):
        """Get notification statistics"""
        try:
            stats = await notification_system.get_notification_statistics(days)

            return {
                "statistics": stats,
                "period_days": days,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Error getting notification statistics", error=str(e))
            raise HTTPException(
                status_code=500, detail=f"Error getting statistics: {str(e)}"
            )

    @router.get("/volcanoes/{volcano_id}/reservoir")
    async def get_reservoir_parameters(
        volcano_id: str = Path(..., description="Volcano identifier")
    ):
        """Get magma reservoir parameter estimates"""
        try:
            logger.info("Reservoir parameters requested", volcano_id=volcano_id)

            # This would gather real monitoring data
            monitoring_data = {
                "seismic": [],
                "deformation": [],
                "geochemical": [],
                "thermal": [],
            }

            # Estimate reservoir parameters
            reservoir_params = await reservoir_estimator.estimate_reservoir_parameters(
                volcano_id, monitoring_data
            )

            response = {
                "volcano_id": reservoir_params.volcano_id,
                "timestamp": reservoir_params.timestamp.isoformat(),
                "location": {
                    "latitude": reservoir_params.location.latitude,
                    "longitude": reservoir_params.location.longitude,
                    "depth": reservoir_params.depth,
                },
                "volume": reservoir_params.volume,
                "volume_change_rate": reservoir_params.volume_change_rate,
                "pressure": reservoir_params.pressure,
                "pressure_change_rate": reservoir_params.pressure_change_rate,
                "temperature": reservoir_params.temperature,
                "composition": reservoir_params.composition,
                "crystallinity": reservoir_params.crystallinity,
                "volatile_content": reservoir_params.volatile_content,
                "connectivity_index": reservoir_params.connectivity_index,
                "connected_reservoirs": reservoir_params.connected_reservoirs,
                "confidence": reservoir_params.estimation_confidence,
                "methods_used": reservoir_params.estimation_methods,
                "uncertainty_bounds": reservoir_params.uncertainty_bounds,
            }

            return response

        except Exception as e:
            logger.error(
                "Error getting reservoir parameters",
                volcano_id=volcano_id,
                error=str(e),
            )
            raise HTTPException(
                status_code=500,
                detail=f"Reservoir parameter estimation failed: {str(e)}",
            )

    @router.get("/system/health")
    async def get_system_health():
        """Get comprehensive system health information"""
        try:
            # Get notification statistics
            notification_stats = await notification_system.get_notification_statistics(
                7
            )  # Last 7 days

            # Get alert statistics for all monitored volcanoes
            volcano_ids = ["yellowstone", "kilauea", "etna", "stromboli", "sakurajima"]
            alert_stats = {}

            for volcano_id in volcano_ids:
                try:
                    stats = await alert_manager.get_alert_statistics(volcano_id, 7)
                    alert_stats[volcano_id] = stats
                except Exception:
                    alert_stats[volcano_id] = {"error": "Unable to get statistics"}

            return {
                "system_status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "uptime": "24h",  # Would be calculated from actual uptime
                "components": {
                    "risk_engine": "healthy",
                    "forecaster": "healthy",
                    "reservoir_estimator": "healthy",
                    "alert_manager": "healthy",
                    "notification_system": "healthy",
                },
                "notification_statistics": notification_stats,
                "alert_statistics": alert_stats,
                "monitored_volcanoes": len(volcano_ids),
                "active_alerts": sum(
                    1
                    for stats in alert_stats.values()
                    if isinstance(stats, dict) and stats.get("total_alerts", 0) > 0
                ),
            }

        except Exception as e:
            logger.error("Error getting system health", error=str(e))
            raise HTTPException(
                status_code=500, detail=f"System health check failed: {str(e)}"
            )

    # Include the routers in the app
    app.include_router(router)
    app.include_router(insar_router)


# Background task functions


async def _send_alert_notification(
    volcano_id: str, alert_level: AlertLevel, notification_system: NotificationSystem
):
    """Background task to send alert notification"""
    try:
        # Create a mock risk assessment for notification
        from models.volcanic_data import VolcanicRiskAssessment

        risk_assessment = VolcanicRiskAssessment(
            volcano_id=volcano_id,
            timestamp=datetime.utcnow(),
            alert_level=alert_level,
            eruption_probability={"24_hours": 0.1, "1_week": 0.2},
            hazard_zones={},
            contributing_factors=["Alert level update"],
            model_ensemble_agreement=0.8,
            composite_risk_score=alert_level.level / 5.0,
        )

        await notification_system.send_alert_notification(volcano_id, risk_assessment)

    except Exception as e:
        logger.error(
            "Error in background alert notification",
            volcano_id=volcano_id,
            error=str(e),
        )


async def _send_custom_notification(
    volcano_id: str,
    message: str,
    priority: str,
    channels: List[str],
    targets: List[str],
    notification_system: NotificationSystem,
):
    """Background task to send custom notification"""
    try:
        # This would use the notification system to send custom messages
        logger.info(
            "Sending custom notification",
            volcano_id=volcano_id,
            message=message,
            priority=priority,
            channels=channels,
            targets=targets,
        )

        # For now, just log the notification
        # In a real implementation, this would integrate with the notification system

    except Exception as e:
        logger.error(
            "Error in background custom notification",
            volcano_id=volcano_id,
            error=str(e),
        )
