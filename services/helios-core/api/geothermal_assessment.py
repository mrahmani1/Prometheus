"""
Geothermal Resource Assessment API

This module provides REST API endpoints for geothermal resource assessment functionality.
Implements thermal modeling, sustainability assessment, and economic viability analysis.

Requirements: 6.2, 6.3
"""

import json
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.geothermal_resource_assessment import (
    EconomicParameters,
    GeothermalAssessmentResult,
    GeothermalReservoir,
    GeothermalResourceAssessmentSystem,
)
from ..core.service_registry import ServiceRegistry
from ..models.geological import GeologicalLocation, ThermalMeasurement

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/geothermal", tags=["geothermal"])

# Global assessment system instance
assessment_system = None

# Pydantic models for API
class GeothermalAssessmentRequest(BaseModel):
    """Request model for geothermal assessment"""

    locations: List[Dict[str, float]] = Field(
        ...,
        description="List of locations with lat, lon, elevation",
        example=[{"latitude": 40.7128, "longitude": -74.0060, "elevation": 10.0}],
    )
    thermal_measurements: Optional[List[Dict[str, Any]]] = Field(
        default=[], description="Available thermal measurements"
    )
    geological_data: Optional[Dict[str, Any]] = Field(
        default={}, description="Geological context data"
    )
    assessment_depth_range: Optional[Tuple[float, float]] = Field(
        default=(1000, 5000), description="Depth range for assessment in meters"
    )
    economic_parameters: Optional[Dict[str, float]] = Field(
        default=None, description="Custom economic parameters"
    )

class GeothermalAssessmentResponse(BaseModel):
    """Response model for geothermal assessment"""

    assessment_id: str
    status: str
    results_count: int
    total_electrical_potential_mw: float
    viable_projects_count: int
    average_viability_score: float
    assessment_timestamp: datetime
    detailed_results: List[Dict[str, Any]]

class EconomicParametersModel(BaseModel):
    """Economic parameters model"""

    drilling_cost_per_meter: float = Field(default=1500.0, description="USD per meter")
    plant_construction_cost_per_mw: float = Field(
        default=4000000.0, description="USD per MW"
    )
    operation_maintenance_cost_annual: float = Field(
        default=0.03, description="Fraction of capital cost"
    )
    electricity_price: float = Field(default=0.08, description="USD per kWh")
    discount_rate: float = Field(default=0.08, description="Annual discount rate")
    project_lifetime_years: int = Field(default=30, description="Project lifetime")
    tax_rate: float = Field(default=0.25, description="Tax rate")

class SustainabilityAnalysisRequest(BaseModel):
    """Request model for sustainability analysis"""

    reservoir_properties: Dict[str, Any]
    extraction_rate: float = Field(description="Extraction rate in kg/s")
    reinjection_rate: float = Field(description="Reinjection rate in kg/s")
    time_horizon_years: int = Field(default=30, description="Analysis time horizon")

class ThermalModelingRequest(BaseModel):
    """Request model for thermal modeling"""

    locations: List[Dict[str, float]]
    depth_profile: List[float] = Field(description="Depth levels for modeling")
    surface_measurements: List[Dict[str, Any]] = Field(default=[])

def get_assessment_system() -> GeothermalResourceAssessmentSystem:
    """Dependency to get assessment system instance"""
    global assessment_system
    if assessment_system is None:
        assessment_system = GeothermalResourceAssessmentSystem()
    return assessment_system

@router.post("/assess", response_model=GeothermalAssessmentResponse)
async def assess_geothermal_potential(
    request: GeothermalAssessmentRequest,
    background_tasks: BackgroundTasks,
    system: GeothermalResourceAssessmentSystem = Depends(get_assessment_system),
):
    """
    Comprehensive geothermal resource assessment

    Performs thermal modeling, sustainability assessment, and economic viability analysis
    for specified locations.
    """
    try:
        logger.info(
            f"Starting geothermal assessment for {len(request.locations)} locations"
        )

        # Convert request data to internal models
        locations = [
            GeologicalLocation(
                latitude=loc["latitude"],
                longitude=loc["longitude"],
                elevation=loc.get("elevation", 0.0),
            )
            for loc in request.locations
        ]

        thermal_measurements = [
            ThermalMeasurement(
                timestamp=datetime.fromisoformat(
                    tm.get("timestamp", datetime.now().isoformat())
                ),
                location=GeologicalLocation(
                    latitude=tm["location"]["latitude"],
                    longitude=tm["location"]["longitude"],
                    elevation=tm["location"].get("elevation", 0.0),
                ),
                temperature=tm["temperature"],
                measurement_method=tm.get("measurement_method", "unknown"),
                spatial_resolution=tm.get("spatial_resolution", 1.0),
                uncertainty=tm.get("uncertainty", 1.0),
            )
            for tm in request.thermal_measurements
        ]

        # Set up economic parameters if provided
        if request.economic_parameters:
            economic_params = EconomicParameters(**request.economic_parameters)
            system.economic_assessor = system.economic_assessor.__class__(
                economic_params
            )

        # Perform assessment
        results = system.assess_geothermal_potential(
            locations=locations,
            thermal_measurements=thermal_measurements,
            geological_data=request.geological_data,
            assessment_depth_range=request.assessment_depth_range,
        )

        # Generate assessment ID
        assessment_id = (
            f"geothermal_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Calculate summary statistics
        total_potential = sum(r.electrical_power_potential for r in results)
        viable_count = sum(1 for r in results if r.economic_viability_score > 0.6)
        avg_viability = (
            sum(r.economic_viability_score for r in results) / len(results)
            if results
            else 0
        )

        # Convert results to response format
        detailed_results = []
        for i, result in enumerate(results):
            detailed_result = {
                "location_id": i + 1,
                "coordinates": {
                    "latitude": result.reservoir.location.latitude,
                    "longitude": result.reservoir.location.longitude,
                    "elevation": result.reservoir.location.elevation,
                },
                "reservoir_properties": {
                    "depth_range_m": result.reservoir.depth_range,
                    "temperature_range_c": result.reservoir.temperature_range,
                    "permeability_md": result.reservoir.permeability,
                    "porosity": result.reservoir.porosity,
                    "volume_m3": result.reservoir.volume,
                    "thermal_conductivity": result.reservoir.thermal_conductivity,
                    "pressure_mpa": result.reservoir.pressure,
                },
                "power_potential": {
                    "thermal_mw": result.thermal_power_potential,
                    "electrical_mw": result.electrical_power_potential,
                    "confidence_interval_mw": result.confidence_interval,
                },
                "sustainability": {
                    "expected_lifetime_years": result.sustainability_years
                },
                "economic_viability": {
                    "viability_score": result.economic_viability_score,
                    "classification": _classify_viability(
                        result.economic_viability_score
                    ),
                },
                "uncertainty_analysis": result.uncertainty_factors,
                "assessment_timestamp": result.assessment_timestamp.isoformat(),
            }
            detailed_results.append(detailed_result)

        # Schedule background report generation
        background_tasks.add_task(_generate_background_report, assessment_id, results)

        response = GeothermalAssessmentResponse(
            assessment_id=assessment_id,
            status="completed",
            results_count=len(results),
            total_electrical_potential_mw=total_potential,
            viable_projects_count=viable_count,
            average_viability_score=avg_viability,
            assessment_timestamp=datetime.now(),
            detailed_results=detailed_results,
        )

        logger.info(f"Geothermal assessment completed: {assessment_id}")
        return response

    except Exception as e:
        logger.error(f"Error in geothermal assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

@router.post("/thermal-modeling")
async def perform_thermal_modeling(
    request: ThermalModelingRequest,
    system: GeothermalResourceAssessmentSystem = Depends(get_assessment_system),
):
    """
    Perform thermal modeling for subsurface temperature estimation
    """
    try:
        logger.info("Starting thermal modeling")

        # Convert request data
        locations = [
            GeologicalLocation(
                latitude=loc["latitude"],
                longitude=loc["longitude"],
                elevation=loc.get("elevation", 0.0),
            )
            for loc in request.locations
        ]

        thermal_measurements = [
            ThermalMeasurement(
                timestamp=datetime.fromisoformat(
                    tm.get("timestamp", datetime.now().isoformat())
                ),
                location=GeologicalLocation(
                    latitude=tm["location"]["latitude"],
                    longitude=tm["location"]["longitude"],
                    elevation=tm["location"].get("elevation", 0.0),
                ),
                temperature=tm["temperature"],
                measurement_method=tm.get("measurement_method", "unknown"),
                spatial_resolution=tm.get("spatial_resolution", 1.0),
                uncertainty=tm.get("uncertainty", 1.0),
            )
            for tm in request.surface_measurements
        ]

        # Perform thermal modeling

        depth_profile = np.array(request.depth_profile)

        temperature_field = system.thermal_engine.estimate_subsurface_temperature(
            locations, depth_profile, thermal_measurements
        )

        # Format response
        response = {
            "status": "completed",
            "temperature_field_shape": temperature_field.shape,
            "depth_profile": request.depth_profile,
            "locations_count": len(locations),
            "temperature_statistics": {
                "min_temperature": float(np.min(temperature_field)),
                "max_temperature": float(np.max(temperature_field)),
                "mean_temperature": float(np.mean(temperature_field)),
                "std_temperature": float(np.std(temperature_field)),
            },
            "modeling_timestamp": datetime.now().isoformat(),
        }

        logger.info("Thermal modeling completed")
        return response

    except Exception as e:
        logger.error(f"Error in thermal modeling: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Thermal modeling failed: {str(e)}"
        )

@router.post("/sustainability-analysis")
async def analyze_sustainability(
    request: SustainabilityAnalysisRequest,
    system: GeothermalResourceAssessmentSystem = Depends(get_assessment_system),
):
    """
    Analyze long-term sustainability of geothermal extraction
    """
    try:
        logger.info("Starting sustainability analysis")

        # Create reservoir from properties
        reservoir_props = request.reservoir_properties
        location = GeologicalLocation(
            latitude=reservoir_props.get("latitude", 0.0),
            longitude=reservoir_props.get("longitude", 0.0),
            elevation=reservoir_props.get("elevation", 0.0),
        )

        reservoir = GeothermalReservoir(
            location=location,
            depth_range=tuple(reservoir_props.get("depth_range", [2000, 2500])),
            temperature_range=tuple(
                reservoir_props.get("temperature_range", [180, 200])
            ),
            permeability=reservoir_props.get("permeability", 100.0),
            porosity=reservoir_props.get("porosity", 0.15),
            fluid_type=reservoir_props.get("fluid_type", "water"),
            salinity=reservoir_props.get("salinity", 5000.0),
            volume=reservoir_props.get("volume", 1e8),
            heat_capacity=reservoir_props.get("heat_capacity", 4200.0),
            thermal_conductivity=reservoir_props.get("thermal_conductivity", 2.5),
            pressure=reservoir_props.get("pressure", 20.0),
            flow_rate_potential=reservoir_props.get("flow_rate_potential", 50.0),
        )

        # Perform sustainability assessment
        sustainability_years, decline_rate = (
            system.sustainability_modeler.assess_sustainability(
                reservoir, request.extraction_rate, request.reinjection_rate
            )
        )

        # Model thermal decline

        extraction_scenario = {
            "decline_rate": decline_rate,
            "extraction_rate": request.extraction_rate,
            "reinjection_rate": request.reinjection_rate,
        }

        thermal_decline = system.sustainability_modeler.model_thermal_decline(
            reservoir, extraction_scenario, request.time_horizon_years
        )

        response = {
            "status": "completed",
            "sustainability_assessment": {
                "expected_lifetime_years": sustainability_years,
                "annual_decline_rate": decline_rate,
                "extraction_rate_kg_s": request.extraction_rate,
                "reinjection_rate_kg_s": request.reinjection_rate,
                "reinjection_ratio": request.reinjection_rate / request.extraction_rate,
            },
            "thermal_decline_profile": {
                "time_years": list(range(request.time_horizon_years + 1)),
                "temperature_c": thermal_decline.tolist(),
                "initial_temperature": thermal_decline[0],
                "final_temperature": thermal_decline[-1],
                "total_decline_percent": (
                    (thermal_decline[0] - thermal_decline[-1]) / thermal_decline[0]
                )
                * 100,
            },
            "sustainability_classification": _classify_sustainability(
                sustainability_years
            ),
            "analysis_timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Sustainability analysis completed: {sustainability_years:.1f} years expected lifetime"
        )
        return response

    except Exception as e:
        logger.error(f"Error in sustainability analysis: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Sustainability analysis failed: {str(e)}"
        )

@router.post("/economic-viability")
async def assess_economic_viability(
    reservoir_properties: Dict[str, Any],
    power_potential_mw: float,
    sustainability_years: float,
    economic_parameters: Optional[EconomicParametersModel] = None,
    system: GeothermalResourceAssessmentSystem = Depends(get_assessment_system),
):
    """
    Assess economic viability of geothermal project
    """
    try:
        logger.info(
            f"Assessing economic viability for {power_potential_mw:.1f} MW project"
        )

        # Set up economic parameters
        if economic_parameters:
            econ_params = EconomicParameters(
                drilling_cost_per_meter=economic_parameters.drilling_cost_per_meter,
                plant_construction_cost_per_mw=economic_parameters.plant_construction_cost_per_mw,
                operation_maintenance_cost_annual=economic_parameters.operation_maintenance_cost_annual,
                electricity_price=economic_parameters.electricity_price,
                discount_rate=economic_parameters.discount_rate,
                project_lifetime_years=economic_parameters.project_lifetime_years,
                tax_rate=economic_parameters.tax_rate,
            )
            economic_assessor = system.economic_assessor.__class__(econ_params)
        else:
            economic_assessor = system.economic_assessor

        # Create reservoir object
        location = GeologicalLocation(
            latitude=reservoir_properties.get("latitude", 0.0),
            longitude=reservoir_properties.get("longitude", 0.0),
            elevation=reservoir_properties.get("elevation", 0.0),
        )

        reservoir = GeothermalReservoir(
            location=location,
            depth_range=tuple(reservoir_properties.get("depth_range", [2000, 2500])),
            temperature_range=tuple(
                reservoir_properties.get("temperature_range", [180, 200])
            ),
            permeability=reservoir_properties.get("permeability", 100.0),
            porosity=reservoir_properties.get("porosity", 0.15),
            fluid_type=reservoir_properties.get("fluid_type", "water"),
            salinity=reservoir_properties.get("salinity", 5000.0),
            volume=reservoir_properties.get("volume", 1e8),
            heat_capacity=reservoir_properties.get("heat_capacity", 4200.0),
            thermal_conductivity=reservoir_properties.get("thermal_conductivity", 2.5),
            pressure=reservoir_properties.get("pressure", 20.0),
            flow_rate_potential=reservoir_properties.get("flow_rate_potential", 50.0),
        )

        # Perform economic assessment
        economic_metrics = economic_assessor.assess_economic_viability(
            reservoir, power_potential_mw, sustainability_years
        )

        response = {
            "status": "completed",
            "economic_metrics": economic_metrics,
            "viability_classification": _classify_viability(
                economic_metrics["viability_score"]
            ),
            "project_parameters": {
                "power_potential_mw": power_potential_mw,
                "sustainability_years": sustainability_years,
                "reservoir_depth_m": sum(reservoir.depth_range) / 2,
            },
            "assessment_timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Economic viability assessment completed: score {economic_metrics['viability_score']:.2f}"
        )
        return response

    except Exception as e:
        logger.error(f"Error in economic viability assessment: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Economic assessment failed: {str(e)}"
        )

@router.get("/assessment/{assessment_id}/report")
async def get_assessment_report(assessment_id: str):
    """
    Retrieve detailed assessment report
    """
    try:
        report_path = f"geothermal_reports/{assessment_id}_report.json"

        try:
            with open(report_path, "r") as f:
                report = json.load(f)
            return report
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Assessment report not found")

    except Exception as e:
        logger.error(f"Error retrieving assessment report: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve report: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "geothermal-assessment",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }

# Helper functions
def _classify_viability(score: float) -> str:
    """Classify economic viability based on score"""
    if score >= 0.8:
        return "Highly Viable"
    elif score >= 0.6:
        return "Viable"
    elif score >= 0.4:
        return "Marginal"
    else:
        return "Not Viable"

def _classify_sustainability(years: float) -> str:
    """Classify sustainability based on expected lifetime"""
    if years >= 50:
        return "Highly Sustainable"
    elif years >= 30:
        return "Sustainable"
    elif years >= 20:
        return "Moderately Sustainable"
    else:
        return "Limited Sustainability"

async def _generate_background_report(
    assessment_id: str, results: List[GeothermalAssessmentResult]
):
    """Generate detailed report in background"""
    try:
        # Create reports directory if it doesn't exist
        import os

        os.makedirs("geothermal_reports", exist_ok=True)

        # Generate comprehensive report
        system = get_assessment_system()
        report_path = f"geothermal_reports/{assessment_id}_report.json"

        system.generate_assessment_report(results, report_path)

        logger.info(f"Background report generated: {report_path}")

    except Exception as e:
        logger.error(f"Error generating background report: {str(e)}")

# Register service with service registry
def register_geothermal_service():
    """Register geothermal assessment service"""
    try:
        registry = ServiceRegistry()
        registry.register_service(
            service_name="geothermal-assessment",
            service_url="http://localhost:8000/api/v1/geothermal",
            health_check_url="http://localhost:8000/api/v1/geothermal/health",
            metadata={
                "version": "1.0.0",
                "capabilities": [
                    "thermal_modeling",
                    "sustainability_analysis",
                    "economic_viability",
                    "uncertainty_quantification",
                ],
            },
        )
        logger.info("Geothermal assessment service registered")
    except Exception as e:
        logger.error(f"Failed to register geothermal service: {str(e)}")

# Initialize service registration on module import
register_geothermal_service()
