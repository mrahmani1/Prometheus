"""
Volcanic Risk Assessment Engine
Implements real-time volcanic risk assessment using multi-source data fusion
"""

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import structlog
import torch.nn as nn
import torch.nn.functional as F
from core.eruption_forecasting import ProbabilisticEruptionForecaster
from core.magma_reservoir import MagmaReservoirEstimator
from models.volcanic_data import (
    AlertLevel,
    DeformationMeasurement,
    GeochemicalSample,
    SeismicEvent,
    ThermalMeasurement,
    VolcanicRiskAssessment,
)

logger = structlog.get_logger()

class VolcanicRiskAssessmentEngine:
    """
    Core engine for real-time volcanic risk assessment

    Integrates multiple data sources and models to provide comprehensive
    volcanic risk assessment with uncertainty quantification
    """

    def __init__(self):
        self.forecaster = ProbabilisticEruptionForecaster()
        self.reservoir_estimator = MagmaReservoirEstimator()

        # Risk assessment parameters
        self.risk_weights = {
            "seismic": 0.25,
            "deformation": 0.25,
            "geochemical": 0.20,
            "thermal": 0.15,
            "historical": 0.15,
        }

        # Alert level thresholds
        self.alert_thresholds = {
            0: {"name": "Normal", "risk_score": 0.0},
            1: {"name": "Advisory", "risk_score": 0.2},
            2: {"name": "Watch", "risk_score": 0.4},
            3: {"name": "Warning", "risk_score": 0.6},
            4: {"name": "Alert", "risk_score": 0.8},
            5: {"name": "Emergency", "risk_score": 0.95},
        }

        logger.info("Volcanic Risk Assessment Engine initialized")

    async def assess_volcanic_risk(self, volcano_id: str) -> VolcanicRiskAssessment:
        """
        Perform comprehensive volcanic risk assessment for a specific volcano

        Args:
            volcano_id: Unique identifier for the volcano

        Returns:
            VolcanicRiskAssessment: Comprehensive risk assessment
        """
        logger.info("Starting volcanic risk assessment", volcano_id=volcano_id)

        try:
            # Gather monitoring data and assess risk components
            monitoring_data = await self._gather_monitoring_data(volcano_id)
            risk_components = await self._assess_all_risk_components(monitoring_data, volcano_id)
            
            # Calculate composite risk and alert level
            composite_risk = self._calculate_composite_risk(risk_components)
            alert_level = self._determine_alert_level(composite_risk)

            # Generate advanced assessments
            advanced_assessments = await self._generate_advanced_assessments(
                volcano_id, monitoring_data, alert_level
            )

            # Create and return final assessment
            return self._create_final_risk_assessment(
                volcano_id, alert_level, composite_risk, 
                risk_components, monitoring_data, advanced_assessments
            )

        except Exception as e:
            logger.error(
                "Error in volcanic risk assessment", volcano_id=volcano_id, error=str(e)
            )
            raise

    async def _assess_all_risk_components(self, monitoring_data: Dict, volcano_id: str) -> Dict[str, float]:
        """Assess all individual risk components."""
        return {
            "seismic": await self._assess_seismic_risk(monitoring_data["seismic"]),
            "deformation": await self._assess_deformation_risk(monitoring_data["deformation"]),
            "geochemical": await self._assess_geochemical_risk(monitoring_data["geochemical"]),
            "thermal": await self._assess_thermal_risk(monitoring_data["thermal"]),
            "historical": await self._assess_historical_risk(volcano_id),
        }

    async def _generate_advanced_assessments(
        self, volcano_id: str, monitoring_data: Dict, alert_level: AlertLevel
    ) -> Dict:
        """Generate advanced assessment components."""
        eruption_forecast = await self.forecaster.generate_probabilistic_forecast(
            volcano_id, monitoring_data
        )
        reservoir_params = await self.reservoir_estimator.estimate_reservoir_parameters(
            volcano_id, monitoring_data
        )
        ensemble_agreement = await self._calculate_ensemble_agreement(volcano_id, monitoring_data)
        hazard_zones = await self._generate_hazard_zones(volcano_id, alert_level, eruption_forecast)

        return {
            "eruption_forecast": eruption_forecast,
            "reservoir_params": reservoir_params,
            "ensemble_agreement": ensemble_agreement,
            "hazard_zones": hazard_zones,
        }

    def _create_final_risk_assessment(
        self, 
        volcano_id: str, 
        alert_level: AlertLevel, 
        composite_risk: float,
        risk_components: Dict[str, float],
        monitoring_data: Dict,
        advanced_assessments: Dict
    ) -> VolcanicRiskAssessment:
        """Create the final volcanic risk assessment."""
        contributing_factors = self._identify_contributing_factors(risk_components)
        eruption_forecast = advanced_assessments["eruption_forecast"]

        risk_assessment = VolcanicRiskAssessment(
            volcano_id=volcano_id,
            timestamp=datetime.utcnow(),
            alert_level=alert_level,
            eruption_probability=eruption_forecast.probability_by_timeframe,
            hazard_zones=advanced_assessments["hazard_zones"],
            contributing_factors=contributing_factors,
            model_ensemble_agreement=advanced_assessments["ensemble_agreement"],
            composite_risk_score=composite_risk,
            reservoir_parameters=advanced_assessments["reservoir_params"],
            confidence_metrics={
                "data_quality": monitoring_data.get("quality_score", 0.8),
                "model_uncertainty": eruption_forecast.uncertainty,
                "ensemble_agreement": advanced_assessments["ensemble_agreement"],
            },
        )

        logger.info(
            "Volcanic risk assessment completed",
            volcano_id=volcano_id,
            alert_level=alert_level.level,
            risk_score=composite_risk,
        )

        return risk_assessment

    async def _gather_monitoring_data(self, volcano_id: str) -> Dict:
        """Gather multi-source monitoring data for the volcano"""
        # This would integrate with real data sources
        # For now, simulate data gathering

        monitoring_data = {
            "seismic": await self._get_recent_seismic_data(volcano_id),
            "deformation": await self._get_recent_deformation_data(volcano_id),
            "geochemical": await self._get_recent_geochemical_data(volcano_id),
            "thermal": await self._get_recent_thermal_data(volcano_id),
            "quality_score": 0.85,  # Overall data quality assessment
        }

        return monitoring_data

    async def _get_recent_seismic_data(self, volcano_id: str) -> List[SeismicEvent]:
        """Get recent seismic events for the volcano"""
        # Simulate seismic data - in reality, this would query the data ingestion service
        return []

    async def _get_recent_deformation_data(
        self, volcano_id: str
    ) -> List[DeformationMeasurement]:
        """Get recent deformation measurements for the volcano"""
        # Simulate deformation data
        return []

    async def _get_recent_geochemical_data(
        self, volcano_id: str
    ) -> List[GeochemicalSample]:
        """Get recent geochemical samples for the volcano"""
        # Simulate geochemical data
        return []

    async def _get_recent_thermal_data(
        self, volcano_id: str
    ) -> List[ThermalMeasurement]:
        """Get recent thermal measurements for the volcano"""
        # Simulate thermal data
        return []

    async def _assess_seismic_risk(self, seismic_data: List[SeismicEvent]) -> float:
        """Assess seismic risk component"""
        if not seismic_data:
            return 0.1  # Baseline risk

        # Analyze seismic patterns
        recent_events = [
            e for e in seismic_data if (datetime.utcnow() - e.timestamp).days <= 7
        ]

        if not recent_events:
            return 0.1

        # Calculate risk based on event frequency, magnitude, and depth
        event_count = len(recent_events)
        avg_magnitude = np.mean([e.magnitude for e in recent_events])
        shallow_events = len([e for e in recent_events if e.depth < 5000])  # < 5km

        # Simple risk scoring (would be more sophisticated in practice)
        frequency_score = min(event_count / 50.0, 1.0)  # Normalize to 50 events/week
        magnitude_score = min(avg_magnitude / 5.0, 1.0)  # Normalize to M5
        depth_score = shallow_events / len(recent_events)

        seismic_risk = frequency_score * 0.4 + magnitude_score * 0.3 + depth_score * 0.3

        return min(seismic_risk, 1.0)

    async def _assess_deformation_risk(
        self, deformation_data: List[DeformationMeasurement]
    ) -> float:
        """Assess deformation risk component"""
        if not deformation_data:
            return 0.1

        # Analyze deformation patterns
        recent_data = [
            d for d in deformation_data if (datetime.utcnow() - d.timestamp).days <= 30
        ]

        if len(recent_data) < 2:
            return 0.1

        # Calculate deformation rate and acceleration
        displacements = [np.linalg.norm(d.displacement_vector) for d in recent_data]
        times = [
            (d.timestamp - recent_data[0].timestamp).total_seconds()
            for d in recent_data
        ]

        if len(displacements) > 1:
            # Simple linear trend analysis
            deformation_rate = (displacements[-1] - displacements[0]) / (
                times[-1] - times[0]
            )
            deformation_risk = min(
                abs(deformation_rate) * 1000, 1.0
            )  # Scale appropriately
        else:
            deformation_risk = 0.1

        return deformation_risk

    async def _assess_geochemical_risk(
        self, geochemical_data: List[GeochemicalSample]
    ) -> float:
        """Assess geochemical risk component"""
        if not geochemical_data:
            return 0.1

        # Analyze gas composition changes
        recent_samples = [
            s for s in geochemical_data if (datetime.utcnow() - s.timestamp).days <= 14
        ]

        if not recent_samples:
            return 0.1

        # Look for indicators like SO2/CO2 ratios, temperature changes
        geochemical_risk = 0.1  # Baseline

        for sample in recent_samples:
            if "SO2" in sample.composition and "CO2" in sample.composition:
                so2_co2_ratio = sample.composition["SO2"] / max(
                    sample.composition["CO2"], 0.001
                )
                if so2_co2_ratio > 0.1:  # Elevated SO2/CO2 indicates magma degassing
                    geochemical_risk += 0.2

            if sample.temperature and sample.temperature > 100:  # Elevated temperature
                geochemical_risk += 0.1

        return min(geochemical_risk, 1.0)

    async def _assess_thermal_risk(
        self, thermal_data: List[ThermalMeasurement]
    ) -> float:
        """Assess thermal risk component"""
        if not thermal_data:
            return 0.1

        # Analyze thermal anomalies
        recent_data = [
            t for t in thermal_data if (datetime.utcnow() - t.timestamp).days <= 7
        ]

        if not recent_data:
            return 0.1

        # Look for temperature increases
        temperatures = [t.temperature for t in recent_data]
        avg_temp = np.mean(temperatures)
        max_temp = np.max(temperatures)

        # Simple thermal risk assessment
        if avg_temp > 50:  # Elevated average temperature
            thermal_risk = 0.3
        else:
            thermal_risk = 0.1

        if max_temp > 100:  # High maximum temperature
            thermal_risk += 0.2

        return min(thermal_risk, 1.0)

    async def _assess_historical_risk(self, volcano_id: str) -> float:
        """Assess historical risk based on volcano's eruption history"""
        # This would query historical eruption database
        # For now, return volcano-specific baseline risk

        historical_risks = {
            "yellowstone": 0.05,  # Low frequency, high impact
            "kilauea": 0.4,  # Frequently active
            "etna": 0.3,  # Regularly active
            "stromboli": 0.5,  # Continuously active
            "sakurajima": 0.4,  # Very active
        }

        return historical_risks.get(volcano_id, 0.2)  # Default moderate risk

    def _calculate_composite_risk(self, risk_components: Dict[str, float]) -> float:
        """Calculate weighted composite risk score"""
        composite_risk = 0.0

        for component, risk_value in risk_components.items():
            weight = self.risk_weights.get(component, 0.0)
            composite_risk += weight * risk_value

        return min(composite_risk, 1.0)

    def _determine_alert_level(self, composite_risk: float) -> AlertLevel:
        """Determine alert level based on composite risk score"""
        alert_level = 0

        for level in sorted(self.alert_thresholds.keys(), reverse=True):
            if composite_risk >= self.alert_thresholds[level]["risk_score"]:
                alert_level = level
                break

        threshold_info = self.alert_thresholds[alert_level]

        return AlertLevel(
            level=alert_level,
            description=threshold_info["name"],
            recommended_actions=self._get_recommended_actions(alert_level),
            validity_period=(
                datetime.utcnow(),
                datetime.utcnow() + timedelta(hours=6),  # 6-hour validity
            ),
            confidence=min(
                0.9, 0.5 + composite_risk * 0.5
            ),  # Higher confidence with higher risk
        )

    def _get_recommended_actions(self, alert_level: int) -> List[str]:
        """Get recommended actions for each alert level"""
        actions = {
            0: ["Continue routine monitoring"],
            1: ["Increase monitoring frequency", "Review evacuation plans"],
            2: [
                "Enhanced monitoring",
                "Prepare emergency services",
                "Public information updates",
            ],
            3: [
                "Continuous monitoring",
                "Evacuate high-risk areas",
                "Close access roads",
            ],
            4: [
                "Emergency response activation",
                "Full evacuation",
                "Aviation warnings",
            ],
            5: [
                "Maximum emergency response",
                "Complete evacuation",
                "International assistance",
            ],
        }

        return actions.get(alert_level, ["Assess situation"])

    def _identify_contributing_factors(
        self, risk_components: Dict[str, float]
    ) -> List[str]:
        """Identify the main contributing factors to the current risk level"""
        factors = []

        for component, risk_value in risk_components.items():
            if risk_value > 0.3:  # Significant contribution
                factors.append(f"Elevated {component} activity")

        if not factors:
            factors.append("Baseline volcanic activity")

        return factors

    async def _calculate_ensemble_agreement(
        self, volcano_id: str, monitoring_data: Dict
    ) -> float:
        """Calculate agreement between different model predictions"""
        # This would run multiple models and compare their outputs
        # For now, simulate ensemble agreement

        # Higher agreement when we have more data
        data_completeness = (
            sum(
                [
                    1 if monitoring_data["seismic"] else 0,
                    1 if monitoring_data["deformation"] else 0,
                    1 if monitoring_data["geochemical"] else 0,
                    1 if monitoring_data["thermal"] else 0,
                ]
            )
            / 4.0
        )

        # Simulate ensemble agreement (would be calculated from actual model outputs)
        base_agreement = 0.7
        agreement = base_agreement + (data_completeness * 0.2)

        return min(agreement, 1.0)

    async def _generate_hazard_zones(
        self, volcano_id: str, alert_level: AlertLevel, eruption_forecast
    ) -> Dict[str, np.ndarray]:
        """Generate hazard zones based on current risk assessment"""
        # This would use sophisticated hazard modeling
        # For now, generate simple concentric zones

        hazard_zones = {}

        if alert_level.level >= 2:  # Watch level or higher
            # Pyroclastic flow zone (high risk within 10km)
            hazard_zones["pyroclastic_flow"] = np.array(
                [[0, 0, 10000]]
            )  # Center + radius

        if alert_level.level >= 3:  # Warning level or higher
            # Ash fall zone (moderate risk within 50km)
            hazard_zones["ash_fall"] = np.array([[0, 0, 50000]])

        if alert_level.level >= 4:  # Alert level or higher
            # Lahar zone (high risk in valleys within 20km)
            hazard_zones["lahar"] = np.array([[0, 0, 20000]])

        return hazard_zones
