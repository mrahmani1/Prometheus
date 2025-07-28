"""
Query Translation from Natural Language to System Parameters
Converts geological queries into structured system parameters
"""

import re
import time
from typing import Dict, List, Any, Union, Optional
import structlog
import json
from datetime import datetime, timedelta

from models.nlp_models import (
    SystemParameter, QueryTranslationResult, QueryIntent, 
    GeologicalEntity, EntityType
)

logger = structlog.get_logger()

class QueryTranslator:
    """
    Translates natural language geological queries into structured system parameters
    that can be used by the Helios AI engine and other system components
    """
    
    def __init__(self):
        self.logger = logger.bind(component="query_translator")
        
        # Initialize parameter mapping rules
        self.parameter_mappings = self._initialize_parameter_mappings()
        
        # Initialize service routing rules
        self.service_routing = self._initialize_service_routing()
        
        # Initialize unit conversion mappings
        self.unit_conversions = self._initialize_unit_conversions()
        
        self.logger.info("Query translator initialized successfully")
    
    def _initialize_parameter_mappings(self) -> Dict[str, Dict]:
        """Initialize mappings from entities to system parameters"""
        return {
            # Location parameters
            "location_mappings": {
                "yellowstone": {
                    "latitude": 44.428,
                    "longitude": -110.588,
                    "region": "yellowstone_caldera",
                    "volcano_type": "caldera"
                },
                "mount st helens": {
                    "latitude": 46.1914,
                    "longitude": -122.1956,
                    "region": "cascades",
                    "volcano_type": "stratovolcano"
                },
                "kilauea": {
                    "latitude": 19.4069,
                    "longitude": -155.2834,
                    "region": "hawaii",
                    "volcano_type": "shield"
                },
                "san andreas": {
                    "fault_type": "transform",
                    "region": "california",
                    "length_km": 1200
                }
            },
            
            # Process parameters
            "process_mappings": {
                "eruption": {
                    "process_type": "volcanic_eruption",
                    "time_scale": "days_to_years",
                    "physics_models": ["magma_dynamics", "gas_expansion", "thermal_evolution"]
                },
                "earthquake": {
                    "process_type": "seismic_rupture",
                    "time_scale": "seconds_to_minutes",
                    "physics_models": ["elastic_wave_propagation", "fault_mechanics"]
                },
                "subduction": {
                    "process_type": "tectonic_subduction",
                    "time_scale": "millions_of_years",
                    "physics_models": ["plate_dynamics", "thermal_evolution", "fluid_flow"]
                },
                "mantle convection": {
                    "process_type": "mantle_convection",
                    "time_scale": "millions_of_years",
                    "physics_models": ["thermal_convection", "viscous_flow", "phase_transitions"]
                }
            },
            
            # Measurement parameters
            "measurement_mappings": {
                "magnitude": {
                    "parameter_name": "earthquake_magnitude",
                    "data_type": "float",
                    "unit": "magnitude_scale",
                    "valid_range": [0.0, 10.0]
                },
                "depth": {
                    "parameter_name": "depth",
                    "data_type": "float",
                    "unit": "kilometers",
                    "valid_range": [0.0, 700.0]
                },
                "temperature": {
                    "parameter_name": "temperature",
                    "data_type": "float",
                    "unit": "celsius",
                    "valid_range": [-50.0, 1500.0]
                },
                "pressure": {
                    "parameter_name": "pressure",
                    "data_type": "float",
                    "unit": "pascals",
                    "valid_range": [0.0, 1e11]
                }
            },
            
            # Time period parameters
            "time_mappings": {
                "holocene": {
                    "time_period": "holocene",
                    "start_years_ago": 11700,
                    "end_years_ago": 0
                },
                "pleistocene": {
                    "time_period": "pleistocene",
                    "start_years_ago": 2580000,
                    "end_years_ago": 11700
                },
                "recent": {
                    "time_period": "recent",
                    "start_years_ago": 100,
                    "end_years_ago": 0
                },
                "historical": {
                    "time_period": "historical",
                    "start_years_ago": 2000,
                    "end_years_ago": 0
                }
            }
        }
    
    def _initialize_service_routing(self) -> Dict[QueryIntent, str]:
        """Initialize routing rules for different query intents"""
        return {
            QueryIntent.SIMULATION_REQUEST: "helios-core",
            QueryIntent.DATA_ANALYSIS: "data-ingestion",
            QueryIntent.MONITORING_QUERY: "volcano-watchtower",
            QueryIntent.HAZARD_ASSESSMENT: "helios-core",
            QueryIntent.RESOURCE_PREDICTION: "helios-core",
            QueryIntent.EXPLANATION_REQUEST: "nlp-service",
            QueryIntent.COMPARISON_QUERY: "helios-core",
            QueryIntent.HISTORICAL_QUERY: "data-ingestion"
        }
    
    def _initialize_unit_conversions(self) -> Dict[str, Dict]:
        """Initialize unit conversion factors"""
        return {
            "length": {
                "m": 1.0,
                "km": 1000.0,
                "cm": 0.01,
                "mm": 0.001,
                "ft": 0.3048,
                "mi": 1609.34
            },
            "temperature": {
                "celsius": {"offset": 0, "scale": 1},
                "kelvin": {"offset": 273.15, "scale": 1},
                "fahrenheit": {"offset": 32, "scale": 1.8}
            },
            "pressure": {
                "pa": 1.0,
                "kpa": 1000.0,
                "mpa": 1e6,
                "gpa": 1e9,
                "bar": 1e5,
                "atm": 101325.0
            },
            "time": {
                "seconds": 1.0,
                "minutes": 60.0,
                "hours": 3600.0,
                "days": 86400.0,
                "years": 31557600.0
            }
        }
    
    def _extract_numerical_values(self, text: str) -> List[Dict[str, Any]]:
        """Extract numerical values with units from text"""
        numerical_values = []
        
        # Pattern for numbers with units
        number_unit_pattern = r'(\d+\.?\d*)\s*([a-zA-Z°]+)'
        matches = re.finditer(number_unit_pattern, text)
        
        for match in matches:
            value = float(match.group(1))
            unit = match.group(2).lower()
            
            numerical_values.append({
                "value": value,
                "unit": unit,
                "text": match.group(0),
                "start": match.start(),
                "end": match.end()
            })
        
        return numerical_values
    
    def _extract_time_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract time references from text"""
        time_references = []
        
        # Relative time patterns
        relative_patterns = [
            (r'(\d+)\s*(years?|months?|days?|hours?)\s*ago', 'relative_past'),
            (r'in\s*(\d+)\s*(years?|months?|days?|hours?)', 'relative_future'),
            (r'next\s*(\d+)\s*(years?|months?|days?|hours?)', 'relative_future'),
            (r'last\s*(\d+)\s*(years?|months?|days?|hours?)', 'relative_past'),
            (r'(recent|recently)', 'recent'),
            (r'(historical|historically)', 'historical'),
            (r'(future|forecast)', 'future')
        ]
        
        for pattern, time_type in relative_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                time_ref = {
                    "type": time_type,
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                }
                
                if match.groups() and len(match.groups()) >= 2:
                    time_ref["value"] = int(match.group(1))
                    time_ref["unit"] = match.group(2)
                
                time_references.append(time_ref)
        
        return time_references
    
    def _translate_location_entities(self, entities: List[GeologicalEntity]) -> List[SystemParameter]:
        """Translate location entities to system parameters"""
        parameters = []
        
        for entity in entities:
            if entity.entity_type != EntityType.LOCATION:
                continue
            
            entity_text = entity.normalized_value or entity.text.lower()
            
            # Check if we have specific mapping for this location
            if entity_text in self.parameter_mappings["location_mappings"]:
                location_data = self.parameter_mappings["location_mappings"][entity_text]
                
                for param_name, param_value in location_data.items():
                    parameter = SystemParameter(
                        name=param_name,
                        value=param_value,
                        data_type=type(param_value).__name__,
                        confidence=entity.confidence,
                        source_text=entity.text
                    )
                    parameters.append(parameter)
            else:
                # Generic location parameter
                parameter = SystemParameter(
                    name="location_name",
                    value=entity.text,
                    data_type="str",
                    confidence=entity.confidence,
                    source_text=entity.text
                )
                parameters.append(parameter)
        
        return parameters
    
    def _translate_process_entities(self, entities: List[GeologicalEntity]) -> List[SystemParameter]:
        """Translate process entities to system parameters"""
        parameters = []
        
        for entity in entities:
            if entity.entity_type != EntityType.PROCESS:
                continue
            
            entity_text = entity.normalized_value or entity.text.lower()
            
            # Check if we have specific mapping for this process
            if entity_text in self.parameter_mappings["process_mappings"]:
                process_data = self.parameter_mappings["process_mappings"][entity_text]
                
                for param_name, param_value in process_data.items():
                    parameter = SystemParameter(
                        name=param_name,
                        value=param_value,
                        data_type=type(param_value).__name__,
                        confidence=entity.confidence,
                        source_text=entity.text
                    )
                    parameters.append(parameter)
            else:
                # Generic process parameter
                parameter = SystemParameter(
                    name="geological_process",
                    value=entity.text,
                    data_type="str",
                    confidence=entity.confidence,
                    source_text=entity.text
                )
                parameters.append(parameter)
        
        return parameters
    
    def _translate_measurement_entities(self, entities: List[GeologicalEntity], 
                                     numerical_values: List[Dict]) -> List[SystemParameter]:
        """Translate measurement entities with numerical values to system parameters"""
        parameters = []
        
        for entity in entities:
            if entity.entity_type != EntityType.MEASUREMENT:
                continue
            
            entity_text = entity.normalized_value or entity.text.lower()
            
            # Find associated numerical values
            associated_values = []
            for num_val in numerical_values:
                # Check if numerical value is near this entity
                if abs(num_val["start"] - entity.start_pos) < 50:  # Within 50 characters
                    associated_values.append(num_val)
            
            if associated_values:
                # Use the closest numerical value
                closest_value = min(associated_values, 
                                  key=lambda x: abs(x["start"] - entity.start_pos))
                
                parameter = SystemParameter(
                    name=entity_text,
                    value=closest_value["value"],
                    data_type="float",
                    confidence=entity.confidence,
                    source_text=f"{entity.text} {closest_value['text']}"
                )
                parameters.append(parameter)
                
                # Add unit parameter if available
                if closest_value["unit"]:
                    unit_parameter = SystemParameter(
                        name=f"{entity_text}_unit",
                        value=closest_value["unit"],
                        data_type="str",
                        confidence=entity.confidence,
                        source_text=closest_value["text"]
                    )
                    parameters.append(unit_parameter)
            else:
                # Measurement entity without specific value
                parameter = SystemParameter(
                    name="measurement_type",
                    value=entity.text,
                    data_type="str",
                    confidence=entity.confidence,
                    source_text=entity.text
                )
                parameters.append(parameter)
        
        return parameters
    
    def _translate_time_entities(self, entities: List[GeologicalEntity], 
                               time_references: List[Dict]) -> List[SystemParameter]:
        """Translate time entities and references to system parameters"""
        parameters = []
        
        # Process geological time period entities
        for entity in entities:
            if entity.entity_type != EntityType.TIME_PERIOD:
                continue
            
            entity_text = entity.normalized_value or entity.text.lower()
            
            if entity_text in self.parameter_mappings["time_mappings"]:
                time_data = self.parameter_mappings["time_mappings"][entity_text]
                
                for param_name, param_value in time_data.items():
                    parameter = SystemParameter(
                        name=param_name,
                        value=param_value,
                        data_type=type(param_value).__name__,
                        confidence=entity.confidence,
                        source_text=entity.text
                    )
                    parameters.append(parameter)
        
        # Process time references from text
        for time_ref in time_references:
            if time_ref["type"] == "relative_past" and "value" in time_ref:
                parameter = SystemParameter(
                    name="time_ago",
                    value=time_ref["value"],
                    data_type="int",
                    confidence=0.8,
                    source_text=time_ref["text"]
                )
                parameters.append(parameter)
                
                unit_parameter = SystemParameter(
                    name="time_unit",
                    value=time_ref["unit"],
                    data_type="str",
                    confidence=0.8,
                    source_text=time_ref["text"]
                )
                parameters.append(unit_parameter)
            
            elif time_ref["type"] == "relative_future" and "value" in time_ref:
                parameter = SystemParameter(
                    name="time_future",
                    value=time_ref["value"],
                    data_type="int",
                    confidence=0.8,
                    source_text=time_ref["text"]
                )
                parameters.append(parameter)
        
        return parameters
    
    def _determine_query_type(self, intent: QueryIntent, entities: List[GeologicalEntity]) -> str:
        """Determine the specific query type based on intent and entities"""
        if intent == QueryIntent.SIMULATION_REQUEST:
            # Check for specific simulation types
            process_entities = [e for e in entities if e.entity_type == EntityType.PROCESS]
            if any("eruption" in e.text.lower() for e in process_entities):
                return "volcanic_simulation"
            elif any("earthquake" in e.text.lower() for e in process_entities):
                return "seismic_simulation"
            elif any("subduction" in e.text.lower() for e in process_entities):
                return "tectonic_simulation"
            else:
                return "general_simulation"
        
        elif intent == QueryIntent.DATA_ANALYSIS:
            return "data_analysis"
        
        elif intent == QueryIntent.MONITORING_QUERY:
            return "real_time_monitoring"
        
        elif intent == QueryIntent.HAZARD_ASSESSMENT:
            return "hazard_assessment"
        
        elif intent == QueryIntent.RESOURCE_PREDICTION:
            return "resource_prediction"
        
        else:
            return "general_query"
    
    async def translate_query(self, query: str, intent: Optional[QueryIntent] = None, 
                            entities: Optional[List[GeologicalEntity]] = None) -> QueryTranslationResult:
        """
        Translate natural language query to system parameters
        
        Args:
            query: Natural language query text
            intent: Classified query intent (optional)
            entities: Extracted entities (optional)
            
        Returns:
            QueryTranslationResult with extracted system parameters
        """
        start_time = time.time()
        
        try:
            self.logger.info("Translating query to system parameters", query=query[:100])
            
            # Extract numerical values and time references
            numerical_values = self._extract_numerical_values(query)
            time_references = self._extract_time_references(query)
            
            parameters = []
            
            # If entities are provided, translate them
            if entities:
                # Translate different entity types
                location_params = self._translate_location_entities(entities)
                parameters.extend(location_params)
                
                process_params = self._translate_process_entities(entities)
                parameters.extend(process_params)
                
                measurement_params = self._translate_measurement_entities(entities, numerical_values)
                parameters.extend(measurement_params)
                
                time_params = self._translate_time_entities(entities, time_references)
                parameters.extend(time_params)
            
            # Add raw numerical values that weren't associated with entities
            for num_val in numerical_values:
                # Check if this value was already processed
                already_processed = any(
                    num_val["text"] in param.source_text for param in parameters 
                    if param.source_text
                )
                
                if not already_processed:
                    parameter = SystemParameter(
                        name="numerical_value",
                        value=num_val["value"],
                        data_type="float",
                        confidence=0.7,
                        source_text=num_val["text"]
                    )
                    parameters.append(parameter)
            
            # Determine query type and target service
            query_type = self._determine_query_type(intent, entities or [])
            target_service = self.service_routing.get(intent) if intent else None
            
            # Calculate overall translation confidence
            if parameters:
                translation_confidence = sum(p.confidence for p in parameters) / len(parameters)
            else:
                translation_confidence = 0.3  # Low confidence if no parameters extracted
            
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.info("Query translation completed", 
                           parameters_count=len(parameters),
                           query_type=query_type,
                           target_service=target_service)
            
            return QueryTranslationResult(
                parameters=parameters,
                query_type=query_type,
                target_service=target_service,
                processing_time_ms=processing_time,
                translation_confidence=translation_confidence
            )
            
        except Exception as e:
            self.logger.error("Query translation failed", error=str(e))
            processing_time = (time.time() - start_time) * 1000
            
            return QueryTranslationResult(
                parameters=[],
                query_type="unknown",
                target_service=None,
                processing_time_ms=processing_time,
                translation_confidence=0.0
            )