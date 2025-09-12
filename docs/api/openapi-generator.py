#!/usr/bin/env python3
"""
OpenAPI Documentation Generator for Prometheus AI Platform
Generates comprehensive, interactive API documentation with live examples
"""

import json
import yaml
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import aiohttp
from pathlib import Path

@dataclass
class APIEndpoint:
    """Represents an API endpoint with full documentation"""
    path: str
    method: str
    summary: str
    description: str
    tags: List[str]
    parameters: List[Dict[str, Any]]
    request_body: Optional[Dict[str, Any]]
    responses: Dict[str, Dict[str, Any]]
    examples: List[Dict[str, Any]]
    security: List[Dict[str, Any]]
    deprecated: bool = False

@dataclass
class APIService:
    """Represents a microservice with its API endpoints"""
    name: str
    title: str
    description: str
    version: str
    base_url: str
    endpoints: List[APIEndpoint]
    schemas: Dict[str, Dict[str, Any]]
    security_schemes: Dict[str, Dict[str, Any]]

class OpenAPIGenerator:
    """Generates comprehensive OpenAPI documentation for all services"""
    
    def __init__(self):
        self.services = {}
        self.global_schemas = {}
        self.security_schemes = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token for authentication"
            },
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for service-to-service communication"
            }
        }
    
    def add_service(self, service: APIService):
        """Add a service to the documentation"""
        self.services[service.name] = service
    
    def generate_helios_core_spec(self) -> APIService:
        """Generate OpenAPI spec for Helios Core service"""
        
        # Define common schemas
        schemas = {
            "GeologicalAnalysisRequest": {
                "type": "object",
                "required": ["data_type", "spatial_extent"],
                "properties": {
                    "data_type": {
                        "type": "string",
                        "enum": ["seismic", "deformation", "geochemical", "thermal", "magnetic"],
                        "description": "Type of geological data to analyze"
                    },
                    "spatial_extent": {
                        "type": "object",
                        "properties": {
                            "min_lat": {"type": "number", "minimum": -90, "maximum": 90},
                            "max_lat": {"type": "number", "minimum": -90, "maximum": 90},
                            "min_lon": {"type": "number", "minimum": -180, "maximum": 180},
                            "max_lon": {"type": "number", "minimum": -180, "maximum": 180}
                        },
                        "required": ["min_lat", "max_lat", "min_lon", "max_lon"]
                    },
                    "temporal_range": {
                        "type": "object",
                        "properties": {
                            "start_time": {"type": "string", "format": "date-time"},
                            "end_time": {"type": "string", "format": "date-time"}
                        }
                    },
                    "parameters": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": "Analysis-specific parameters"
                    }
                }
            },
            "AnalysisResponse": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["success", "error", "processing"]},
                    "results": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": "Analysis results"
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "processing_time": {"type": "number"},
                            "model_version": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    },
                    "correlation_id": {"type": "string"}
                }
            },
            "SimulationRequest": {
                "type": "object",
                "required": ["simulation_type", "initial_conditions", "boundary_conditions", "time_span"],
                "properties": {
                    "simulation_type": {
                        "type": "string",
                        "enum": ["thermal", "mechanical", "electromagnetic", "coupled"],
                        "description": "Type of physics simulation"
                    },
                    "initial_conditions": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": "Initial conditions for simulation"
                    },
                    "boundary_conditions": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": "Boundary conditions for simulation"
                    },
                    "time_span": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Simulation time span in seconds"
                    },
                    "parameters": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": "Simulation parameters"
                    }
                }
            },
            "ErrorResponse": {
                "type": "object",
                "properties": {
                    "error": {"type": "string"},
                    "status_code": {"type": "integer"},
                    "correlation_id": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"}
                }
            }
        }
        
        # Define endpoints
        endpoints = [
            APIEndpoint(
                path="/health",
                method="GET",
                summary="Health Check",
                description="Check service health and status",
                tags=["Health"],
                parameters=[],
                request_body=None,
                responses={
                    "200": {
                        "description": "Service is healthy",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "timestamp": {"type": "string"},
                                        "version": {"type": "string"},
                                        "modules_loaded": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            }
                        }
                    }
                },
                examples=[
                    {
                        "name": "Health Check Example",
                        "description": "Basic health check",
                        "request": {},
                        "response": {
                            "status": "healthy",
                            "timestamp": "2025-01-09T10:00:00Z",
                            "version": "1.0.0",
                            "modules_loaded": ["GeologicalPINNEngine", "MultiPhysicsCouplingEngine"]
                        }
                    }
                ],
                security=[]
            ),
            APIEndpoint(
                path="/analyze/geological",
                method="POST",
                summary="Geological Data Analysis",
                description="Perform advanced geological data analysis using Physics-Informed Neural Networks",
                tags=["Analysis", "Geological"],
                parameters=[
                    {
                        "name": "X-Correlation-ID",
                        "in": "header",
                        "description": "Request correlation ID for tracking",
                        "schema": {"type": "string"}
                    }
                ],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/GeologicalAnalysisRequest"}
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "Analysis completed successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AnalysisResponse"}
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid request parameters",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    },
                    "500": {
                        "description": "Internal server error",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    }
                },
                examples=[
                    {
                        "name": "Seismic Analysis Example",
                        "description": "Analyze seismic data for volcanic region",
                        "request": {
                            "data_type": "seismic",
                            "spatial_extent": {
                                "min_lat": 45.0,
                                "max_lat": 46.0,
                                "min_lon": -122.0,
                                "max_lon": -121.0
                            },
                            "temporal_range": {
                                "start_time": "2025-01-01T00:00:00Z",
                                "end_time": "2025-01-09T00:00:00Z"
                            },
                            "parameters": {
                                "frequency_range": [1.0, 10.0],
                                "magnitude_threshold": 2.0
                            }
                        },
                        "response": {
                            "status": "success",
                            "results": {
                                "event_count": 15,
                                "magnitude_distribution": [2.1, 2.3, 2.8, 3.1],
                                "depth_analysis": {"mean_depth": 8.5, "std_depth": 2.1}
                            },
                            "metadata": {
                                "processing_time": 2.34,
                                "model_version": "pinn-v1.2.0",
                                "confidence": 0.87
                            }
                        }
                    }
                ],
                security=[{"BearerAuth": []}]
            ),
            APIEndpoint(
                path="/simulate/physics",
                method="POST",
                summary="Physics Simulation",
                description="Run physics-informed simulations with geological constraints",
                tags=["Simulation", "Physics"],
                parameters=[
                    {
                        "name": "X-Correlation-ID",
                        "in": "header",
                        "description": "Request correlation ID for tracking",
                        "schema": {"type": "string"}
                    }
                ],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/SimulationRequest"}
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "Simulation completed successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AnalysisResponse"}
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid simulation parameters",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    }
                },
                examples=[
                    {
                        "name": "Thermal Simulation Example",
                        "description": "Simulate thermal evolution of magma chamber",
                        "request": {
                            "simulation_type": "thermal",
                            "initial_conditions": {
                                "temperature": 1200.0,
                                "pressure": 200e6,
                                "composition": "basaltic"
                            },
                            "boundary_conditions": {
                                "surface_temperature": 15.0,
                                "heat_flux": 0.1
                            },
                            "time_span": 86400.0,
                            "parameters": {
                                "mesh_resolution": 100,
                                "time_step": 60.0
                            }
                        },
                        "response": {
                            "status": "success",
                            "results": {
                                "final_temperature": 1150.0,
                                "cooling_rate": 0.0005,
                                "crystallization_fraction": 0.15
                            },
                            "metadata": {
                                "simulation_time": 45.2,
                                "convergence": True,
                                "iterations": 1440
                            }
                        }
                    }
                ],
                security=[{"BearerAuth": []}]
            )
        ]
        
        return APIService(
            name="helios-core",
            title="Helios Core Service",
            description="Physics-Informed Neural Network engine for geological analysis and simulation",
            version="1.0.0",
            base_url="http://localhost:8001",
            endpoints=endpoints,
            schemas=schemas,
            security_schemes=self.security_schemes
        )
    
    def generate_volcano_watchtower_spec(self) -> APIService:
        """Generate OpenAPI spec for Volcano Watchtower service"""
        
        schemas = {
            "VolcanicMonitoringRequest": {
                "type": "object",
                "required": ["volcano_id", "data_sources"],
                "properties": {
                    "volcano_id": {"type": "string", "description": "Unique volcano identifier"},
                    "data_sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["seismic", "gps", "insar", "gas", "thermal"]},
                        "description": "Data sources to monitor"
                    },
                    "alert_thresholds": {
                        "type": "object",
                        "properties": {
                            "magnitude_threshold": {"type": "number"},
                            "deformation_threshold": {"type": "number"},
                            "gas_threshold": {"type": "number"}
                        }
                    }
                }
            },
            "EruptionForecast": {
                "type": "object",
                "properties": {
                    "volcano_id": {"type": "string"},
                    "probability": {"type": "number", "minimum": 0, "maximum": 1},
                    "time_window": {"type": "string"},
                    "confidence_interval": {
                        "type": "object",
                        "properties": {
                            "lower": {"type": "number"},
                            "upper": {"type": "number"}
                        }
                    },
                    "contributing_factors": {"type": "array", "items": {"type": "string"}},
                    "alert_level": {"type": "string", "enum": ["green", "yellow", "orange", "red"]}
                }
            }
        }
        
        endpoints = [
            APIEndpoint(
                path="/monitor/volcano/{volcano_id}",
                method="POST",
                summary="Start Volcanic Monitoring",
                description="Initialize real-time monitoring for a specific volcano",
                tags=["Monitoring", "Volcanic"],
                parameters=[
                    {
                        "name": "volcano_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Unique volcano identifier"
                    }
                ],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/VolcanicMonitoringRequest"}
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "Monitoring started successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "monitoring_id": {"type": "string"},
                                        "data_sources": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            }
                        }
                    }
                },
                examples=[
                    {
                        "name": "Mount St. Helens Monitoring",
                        "description": "Start comprehensive monitoring for Mount St. Helens",
                        "request": {
                            "volcano_id": "MSH",
                            "data_sources": ["seismic", "gps", "insar", "gas"],
                            "alert_thresholds": {
                                "magnitude_threshold": 2.0,
                                "deformation_threshold": 0.01,
                                "gas_threshold": 100.0
                            }
                        },
                        "response": {
                            "status": "monitoring_active",
                            "monitoring_id": "MSH-2025-001",
                            "data_sources": ["seismic", "gps", "insar", "gas"]
                        }
                    }
                ],
                security=[{"BearerAuth": []}]
            ),
            APIEndpoint(
                path="/forecast/eruption/{volcano_id}",
                method="GET",
                summary="Get Eruption Forecast",
                description="Get probabilistic eruption forecast for a volcano",
                tags=["Forecasting", "Volcanic"],
                parameters=[
                    {
                        "name": "volcano_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Unique volcano identifier"
                    },
                    {
                        "name": "time_horizon",
                        "in": "query",
                        "schema": {"type": "string", "enum": ["24h", "7d", "30d", "1y"]},
                        "description": "Forecast time horizon"
                    }
                ],
                request_body=None,
                responses={
                    "200": {
                        "description": "Forecast generated successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/EruptionForecast"}
                            }
                        }
                    }
                },
                examples=[
                    {
                        "name": "Yellowstone Forecast",
                        "description": "Get 30-day eruption forecast for Yellowstone",
                        "request": {},
                        "response": {
                            "volcano_id": "YEL",
                            "probability": 0.001,
                            "time_window": "30d",
                            "confidence_interval": {"lower": 0.0005, "upper": 0.002},
                            "contributing_factors": ["seismic_swarm", "ground_deformation"],
                            "alert_level": "yellow"
                        }
                    }
                ],
                security=[{"BearerAuth": []}]
            )
        ]
        
        return APIService(
            name="volcano-watchtower",
            title="Volcano Watchtower Service",
            description="Real-time volcanic monitoring and eruption forecasting system",
            version="1.0.0",
            base_url="http://localhost:8002",
            endpoints=endpoints,
            schemas=schemas,
            security_schemes=self.security_schemes
        )
    
    def generate_openapi_spec(self, service_name: str) -> Dict[str, Any]:
        """Generate complete OpenAPI 3.0 specification for a service"""
        
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not found")
        
        service = self.services[service_name]
        
        # Build paths
        paths = {}
        for endpoint in service.endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}
            
            operation = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "operationId": f"{endpoint.method.lower()}_{endpoint.path.replace('/', '_').replace('{', '').replace('}', '')}",
                "parameters": endpoint.parameters,
                "responses": endpoint.responses
            }
            
            if endpoint.request_body:
                operation["requestBody"] = endpoint.request_body
            
            if endpoint.security:
                operation["security"] = endpoint.security
            
            if endpoint.deprecated:
                operation["deprecated"] = True
            
            # Add examples
            if endpoint.examples:
                operation["x-examples"] = endpoint.examples
            
            paths[endpoint.path][endpoint.method.lower()] = operation
        
        # Build complete spec
        spec = {
            "openapi": "3.0.3",
            "info": {
                "title": service.title,
                "description": service.description,
                "version": service.version,
                "contact": {
                    "name": "Prometheus AI Platform Team",
                    "email": "support@prometheus-ai.com",
                    "url": "https://prometheus-ai.com/support"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": service.base_url,
                    "description": "Development server"
                },
                {
                    "url": service.base_url.replace("localhost", "staging.prometheus-ai.com"),
                    "description": "Staging server"
                },
                {
                    "url": service.base_url.replace("localhost", "api.prometheus-ai.com"),
                    "description": "Production server"
                }
            ],
            "paths": paths,
            "components": {
                "schemas": service.schemas,
                "securitySchemes": service.security_schemes,
                "parameters": {
                    "CorrelationId": {
                        "name": "X-Correlation-ID",
                        "in": "header",
                        "description": "Request correlation ID for tracking",
                        "schema": {"type": "string"},
                        "example": "req-12345-abcde"
                    }
                },
                "responses": {
                    "UnauthorizedError": {
                        "description": "Authentication information is missing or invalid",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    },
                    "ForbiddenError": {
                        "description": "Insufficient permissions",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    }
                }
            },
            "security": [
                {"BearerAuth": []}
            ],
            "tags": [
                {"name": "Health", "description": "Health check endpoints"},
                {"name": "Analysis", "description": "Data analysis endpoints"},
                {"name": "Simulation", "description": "Physics simulation endpoints"},
                {"name": "Monitoring", "description": "Real-time monitoring endpoints"},
                {"name": "Forecasting", "description": "Prediction and forecasting endpoints"}
            ]
        }
        
        return spec
    
    def generate_all_specs(self) -> Dict[str, Dict[str, Any]]:
        """Generate OpenAPI specs for all services"""
        
        # Add all services
        self.add_service(self.generate_helios_core_spec())
        self.add_service(self.generate_volcano_watchtower_spec())
        
        # Generate specs
        specs = {}
        for service_name in self.services:
            specs[service_name] = self.generate_openapi_spec(service_name)
        
        return specs
    
    def save_specs(self, output_dir: str = "docs/api/specs"):
        """Save all OpenAPI specs to files"""
        
        specs = self.generate_all_specs()
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for service_name, spec in specs.items():
            # Save as JSON
            json_path = Path(output_dir) / f"{service_name}.json"
            with open(json_path, 'w') as f:
                json.dump(spec, f, indent=2)
            
            # Save as YAML
            yaml_path = Path(output_dir) / f"{service_name}.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
        
        print(f"Generated OpenAPI specs for {len(specs)} services in {output_dir}")

if __name__ == "__main__":
    generator = OpenAPIGenerator()
    generator.save_specs()