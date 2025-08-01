"""
Conversational Response Generation for Geological Explanations
Generates natural language responses for geological queries and explanations
"""

import random
import signal
import time
from typing import Dict, List, Optional

import structlog
import torch.nn as nn
import torch.nn.functional as F
from models.nlp_models import (
    ConversationalResponse,
    EntityType,
    GeologicalEntity,
    QueryIntent,
)
# Neural network imports handled in model files

logger = structlog.get_logger()

class ConversationalResponseGenerator:
    """
    Generates conversational responses for geological explanations and queries
    Uses template-based generation with geological knowledge integration
    """

    def __init__(self):
        self.logger = logger.bind(component="response_generator")

        # Initialize geological knowledge base
        self.geological_knowledge = self._initialize_geological_knowledge()

        # Initialize response templates
        self.response_templates = self._initialize_response_templates()

        # Initialize follow-up suggestions
        self.follow_up_templates = self._initialize_follow_up_templates()

        self.logger.info("Conversational response generator initialized successfully")

    def _initialize_geological_knowledge(self) -> Dict[str, Dict]:
        """Initialize geological knowledge base for explanations"""
        return {
            "volcanic_processes": {
                "eruption": {
                    "definition": "A volcanic eruption occurs when magma, gas, and other materials are expelled from a volcano's vent or fissure.",
                    "mechanism": "Eruptions are driven by the buoyancy of magma, gas expansion, and pressure differences between the magma chamber and surface.",
                    "types": ["explosive", "effusive", "phreatomagmatic"],
                    "factors": [
                        "magma composition",
                        "gas content",
                        "vent geometry",
                        "external water",
                    ],
                    "examples": [
                        "Mount St. Helens 1980",
                        "Kilauea continuous eruption",
                        "Pinatubo 1991",
                    ],
                },
                "magma": {
                    "definition": "Magma is molten rock beneath the Earth's surface, composed of liquid rock, crystals, and dissolved gases.",
                    "formation": "Magma forms through partial melting of rocks in the mantle or crust due to increased temperature, decreased pressure, or addition of volatiles.",
                    "composition": ["basaltic", "andesitic", "dacitic", "rhyolitic"],
                    "properties": [
                        "temperature",
                        "viscosity",
                        "gas content",
                        "crystal content",
                    ],
                    "evolution": "Magma evolves through fractional crystallization, assimilation, and mixing processes.",
                },
                "volcano": {
                    "definition": "A volcano is a vent or fissure in Earth's surface through which magma, gas, and pyroclastic materials are erupted.",
                    "types": [
                        "stratovolcano",
                        "shield volcano",
                        "cinder cone",
                        "caldera",
                    ],
                    "formation": "Volcanoes form at plate boundaries, hotspots, and areas of crustal weakness where magma can reach the surface.",
                    "components": [
                        "magma chamber",
                        "conduit",
                        "vent",
                        "crater",
                        "volcanic edifice",
                    ],
                    "hazards": [
                        "lava flows",
                        "pyroclastic flows",
                        "ash fall",
                        "lahars",
                        "volcanic gases",
                    ],
                },
            },
            "seismic_processes": {
                "earthquake": {
                    "definition": "An earthquake is the sudden release of energy stored in rocks, causing ground shaking and seismic waves.",
                    "mechanism": "Earthquakes occur when stress accumulated along faults exceeds the strength of rocks, causing sudden rupture and slip.",
                    "types": ["tectonic", "volcanic", "induced", "collapse"],
                    "measurement": [
                        "magnitude",
                        "intensity",
                        "depth",
                        "focal mechanism",
                    ],
                    "effects": [
                        "ground shaking",
                        "surface rupture",
                        "liquefaction",
                        "landslides",
                        "tsunamis",
                    ],
                },
                "fault": {
                    "definition": "A fault is a fracture in rock along which movement has occurred, accommodating tectonic stress.",
                    "types": ["normal", "reverse", "strike-slip", "oblique"],
                    "formation": "Faults form when tectonic stress exceeds rock strength, creating zones of weakness and deformation.",
                    "activity": ["active", "inactive", "potentially active"],
                    "hazards": [
                        "surface rupture",
                        "earthquake generation",
                        "ground deformation",
                    ],
                },
            },
            "tectonic_processes": {
                "plate_tectonics": {
                    "definition": "Plate tectonics is the theory describing the movement of Earth's lithospheric plates over the asthenosphere.",
                    "mechanism": "Plate motion is driven by mantle convection, ridge push, slab pull, and basal drag forces.",
                    "boundaries": ["divergent", "convergent", "transform"],
                    "evidence": [
                        "seafloor spreading",
                        "magnetic reversals",
                        "earthquake distribution",
                        "volcanic activity",
                    ],
                    "rates": "Plates move at rates of 2-10 cm per year, similar to fingernail growth.",
                },
                "subduction": {
                    "definition": "Subduction is the process where one tectonic plate descends beneath another into the mantle.",
                    "mechanism": "Dense oceanic lithosphere sinks into the mantle due to negative buoyancy, pulling the plate behind it.",
                    "features": [
                        "trench",
                        "volcanic arc",
                        "accretionary wedge",
                        "forearc",
                        "backarc",
                    ],
                    "processes": [
                        "dehydration",
                        "melting",
                        "metamorphism",
                        "seismicity",
                    ],
                    "examples": [
                        "Cascadia",
                        "Japan Trench",
                        "Peru-Chile Trench",
                        "Aleutian Trench",
                    ],
                },
            },
            "geological_materials": {
                "basalt": {
                    "definition": "Basalt is a fine-grained, dark-colored volcanic rock composed primarily of plagioclase and pyroxene.",
                    "formation": "Basalt forms from the rapid cooling of low-silica magma at or near Earth's surface.",
                    "composition": "Basalt is mafic, containing 45-52% silica and rich in iron and magnesium.",
                    "occurrence": [
                        "oceanic crust",
                        "volcanic islands",
                        "continental flood basalts",
                        "mid-ocean ridges",
                    ],
                    "properties": [
                        "high density",
                        "low viscosity when molten",
                        "dark color",
                        "fine-grained texture",
                    ],
                },
                "granite": {
                    "definition": "Granite is a coarse-grained, light-colored intrusive igneous rock composed of quartz, feldspar, and mica.",
                    "formation": "Granite forms from the slow cooling of high-silica magma deep within Earth's crust.",
                    "composition": "Granite is felsic, containing 65-75% silica and rich in aluminum and potassium.",
                    "occurrence": [
                        "continental crust",
                        "batholiths",
                        "plutons",
                        "mountain cores",
                    ],
                    "properties": [
                        "low density",
                        "high viscosity when molten",
                        "light color",
                        "coarse-grained texture",
                    ],
                },
            },
            "geological_time": {
                "deep_time": {
                    "definition": "Deep time refers to the vast timescales of geological processes, spanning millions to billions of years.",
                    "concept": "Geological processes operate on timescales far beyond human experience, requiring different perspectives on change.",
                    "examples": [
                        "mountain building (millions of years)",
                        "continental drift (hundreds of millions)",
                        "Earth formation (4.6 billion years)",
                    ],
                    "importance": "Understanding deep time is crucial for comprehending geological processes and Earth's evolution.",
                },
                "geological_time_scale": {
                    "definition": "The geological time scale divides Earth's history into hierarchical units based on significant events and fossil records.",
                    "divisions": ["eons", "eras", "periods", "epochs", "ages"],
                    "current": "We are currently in the Cenozoic Era, Quaternary Period, Holocene Epoch.",
                    "boundaries": "Time boundaries are defined by major geological or biological events, such as mass extinctions.",
                },
            },
        }

    def _initialize_response_templates(self) -> Dict[str, List[str]]:
        """Initialize response templates for different query types"""
        return {
            "explanation": [
                "{definition} {mechanism}",
                "Let me explain {topic}. {definition} {additional_info}",
                "{topic} is {definition} The process works as follows: {mechanism}",
                "To understand {topic}, it's important to know that {definition} {context}",
            ],
            "comparison": [
                "The main differences between {item1} and {item2} are: {differences}",
                "While {item1} and {item2} are both {category}, they differ in {key_differences}",
                "Comparing {item1} to {item2}: {comparison_details}",
            ],
            "process_description": [
                "The {process} process occurs through the following steps: {steps}",
                "{process} is a {type} process that {description}. The key factors involved are {factors}",
                "Understanding {process} requires considering {key_aspects}",
            ],
            "hazard_explanation": [
                "The hazards associated with {phenomenon} include {hazards}. {risk_context}",
                "{phenomenon} poses several risks: {hazards}. The severity depends on {factors}",
                "From a hazard perspective, {phenomenon} can cause {primary_hazards} and {secondary_hazards}",
            ],
            "monitoring_explanation": [
                "We monitor {phenomenon} using {instruments} to detect {signals}",
                "Real-time monitoring of {phenomenon} involves {monitoring_methods} and {data_analysis}",
                "The current monitoring data shows {current_status} with {confidence_level} confidence",
            ],
            "simulation_explanation": [
                "To simulate {process}, we use {models} that incorporate {physics_constraints}",
                "Our simulation approach for {process} combines {methods} to predict {outcomes}",
                "The {process} simulation considers {parameters} over {timescales}",
            ],
        }

    def _initialize_follow_up_templates(self) -> Dict[str, List[str]]:
        """Initialize follow-up question templates"""
        return {
            "explanation": [
                "Would you like to know more about {related_topic}?",
                "Are you interested in learning about {related_process}?",
                "Would you like me to explain {related_concept} as well?",
                "Do you have questions about how {topic} relates to {related_topic}?",
            ],
            "comparison": [
                "Would you like to compare {topic} with other {category}?",
                "Are you interested in more details about {specific_aspect}?",
                "Would you like to know about {related_comparison}?",
            ],
            "monitoring": [
                "Would you like to see historical data for {location}?",
                "Are you interested in setting up alerts for {phenomenon}?",
                "Would you like to know about other monitoring parameters?",
            ],
            "simulation": [
                "Would you like to run a simulation with different parameters?",
                "Are you interested in exploring {alternative_scenario}?",
                "Would you like to see uncertainty estimates for this simulation?",
            ],
        }

    def _get_geological_knowledge(self, topic: str) -> Optional[Dict]:
        """Retrieve geological knowledge for a specific topic"""
        topic_lower = topic.lower()

        # Search through all knowledge categories
        for category, topics in self.geological_knowledge.items():
            for key, knowledge in topics.items():
                if key in topic_lower or topic_lower in key:
                    return knowledge

                # Check synonyms or related terms
                if "types" in knowledge:
                    for type_name in knowledge["types"]:
                        if type_name in topic_lower:
                            return knowledge

        return None

    def _generate_explanation_response(
        self,
        query: str,
        entities: List[GeologicalEntity],
        context: Optional[Dict] = None,
    ) -> str:
        """Generate explanation response for geological concepts"""
        # Find the main topic from entities
        main_topic = None
        for entity in entities:
            if entity.entity_type in [
                EntityType.PROCESS,
                EntityType.MATERIAL,
                EntityType.LOCATION,
            ]:
                main_topic = entity.text.lower()
                break

        if not main_topic:
            # Extract topic from query
            explanation_keywords = [
                "what is",
                "explain",
                "describe",
                "tell me about",
                "how does",
                "why does",
            ]
            for keyword in explanation_keywords:
                if keyword in query.lower():
                    # Extract the topic after the keyword
                    parts = query.lower().split(keyword)
                    if len(parts) > 1:
                        main_topic = (
                            parts[1].strip().split()[0:3]
                        )  # Take first few words
                        main_topic = " ".join(main_topic).strip("?.,")
                        break

        if main_topic:
            knowledge = self._get_geological_knowledge(main_topic)
            if knowledge:
                template = random.choice(self.response_templates["explanation"])

                response = template.format(
                    topic=main_topic.title(),
                    definition=knowledge.get("definition", ""),
                    mechanism=knowledge.get("mechanism", ""),
                    additional_info=knowledge.get(
                        "formation", knowledge.get("process", "")
                    ),
                    context=knowledge.get("importance", ""),
                )

                # Add examples if available
                if "examples" in knowledge:
                    examples = ", ".join(
                        knowledge["examples"][:2]
                    )  # Limit to 2 examples
                    response += f" Examples include {examples}."

                return response.strip()

        # Fallback response
        return (
            "I'd be happy to explain geological concepts. Could you please specify which "
            "geological process, feature, or phenomenon you'd like to learn about?"
        )

    def _generate_comparison_response(
        self, query: str, entities: List[GeologicalEntity]
    ) -> str:
        """Generate comparison response for geological features"""
        # Extract items being compared
        comparison_items = []
        for entity in entities:
            if entity.entity_type in [
                EntityType.PROCESS,
                EntityType.MATERIAL,
                EntityType.LOCATION,
            ]:
                comparison_items.append(entity.text)

        if len(comparison_items) >= 2:
            item1, item2 = comparison_items[0], comparison_items[1]

            # Get knowledge for both items
            knowledge1 = self._get_geological_knowledge(item1)
            knowledge2 = self._get_geological_knowledge(item2)

            if knowledge1 and knowledge2:
                differences = []

                # Compare definitions
                if "definition" in knowledge1 and "definition" in knowledge2:
                    differences.append(
                        f"{item1} is {knowledge1['definition'].lower()}, "
                        f"while {item2} is {knowledge2['definition'].lower()}"
                    )

                # Compare formation processes
                if "formation" in knowledge1 and "formation" in knowledge2:
                    differences.append(
                        f"Formation: {knowledge1['formation']} vs {knowledge2['formation']}"
                    )

                # Compare properties
                if "properties" in knowledge1 and "properties" in knowledge2:
                    props1 = ", ".join(knowledge1["properties"][:3])
                    props2 = ", ".join(knowledge2["properties"][:3])
                    differences.append(
                        f"Key properties: {item1} has {props1}, "
                        f"while {item2} has {props2}"
                    )

                if differences:
                    template = random.choice(self.response_templates["comparison"])
                    return template.format(
                        item1=item1,
                        item2=item2,
                        category="geological features",
                        differences=". ".join(
                            differences[:2]
                        ),  # Limit to 2 differences
                        key_differences=(
                            differences[0]
                            if differences
                            else "their formation processes"
                        ),
                        comparison_details=". ".join(differences),
                    )

        return (
            "To provide a meaningful comparison, please specify which geological features, "
            "processes, or materials you'd like me to compare."
        )

    def _generate_monitoring_response(
        self,
        query: str,
        entities: List[GeologicalEntity],
        geological_data: Optional[Dict] = None,
    ) -> str:
        """Generate response for monitoring queries"""
        # Find location or phenomenon being monitored
        location = None
        phenomenon = None

        for entity in entities:
            if entity.entity_type == EntityType.LOCATION:
                location = entity.text
            elif entity.entity_type == EntityType.PROCESS:
                phenomenon = entity.text

        if geological_data and "current_status" in geological_data:
            status = geological_data["current_status"]
            template = random.choice(self.response_templates["monitoring_explanation"])

            return template.format(
                phenomenon=phenomenon or "volcanic activity",
                instruments="seismometers, GPS stations, and gas sensors",
                signals="ground deformation, seismic activity, and gas emissions",
                monitoring_methods="real-time sensor networks",
                data_analysis="automated pattern recognition",
                current_status=status,
                confidence_level="high",
            )
        else:
            return (
                f"Current monitoring shows normal background levels for {location or 'the region'}. "
                "We continuously track seismic activity, ground deformation, and gas emissions "
                "using a network of sensitive instruments."
            )

    def _generate_simulation_response(
        self, query: str, entities: List[GeologicalEntity]
    ) -> str:
        """Generate response for simulation requests"""
        # Find process to simulate
        process = None
        for entity in entities:
            if entity.entity_type == EntityType.PROCESS:
                process = entity.text
                break

        if process:
            template = random.choice(self.response_templates["simulation_explanation"])

            # Determine simulation details based on process
            if "eruption" in process.lower():
                models = "Physics-Informed Neural Networks (PINNs)"
                physics = "magma dynamics, gas expansion, and thermal evolution"
                outcomes = "eruption probability, ash dispersion, and lava flow paths"
                timescales = "days to years"
            elif "earthquake" in process.lower():
                models = "seismic wave propagation models"
                physics = "elastic wave equations and fault mechanics"
                outcomes = "ground shaking intensity and duration"
                timescales = "seconds to minutes"
            else:
                models = "multi-physics simulation models"
                physics = "fundamental physical laws"
                outcomes = "system evolution and behavior"
                timescales = "appropriate geological timescales"

            return template.format(
                process=process,
                models=models,
                physics_constraints=physics,
                methods="advanced numerical methods and machine learning",
                outcomes=outcomes,
                parameters="initial conditions, material properties, and boundary conditions",
                timescales=timescales,
            )

        return (
            "I can help you set up geological simulations. Please specify which process "
            "you'd like to simulate (e.g., volcanic eruption, earthquake, or tectonic evolution)."
        )

    def _generate_follow_up_suggestions(
        self, response_type: str, main_topic: str
    ) -> List[str]:
        """Generate follow-up question suggestions"""
        if response_type not in self.follow_up_templates:
            return []

        templates = self.follow_up_templates[response_type]
        suggestions = []

        # Generate 2-3 follow-up suggestions
        for template in templates[:3]:
            if response_type == "explanation":
                related_topics = {
                    "volcano": "volcanic hazards",
                    "earthquake": "seismic waves",
                    "magma": "volcanic eruptions",
                    "fault": "earthquake mechanics",
                    "plate tectonics": "continental drift",
                }
                related_topic = related_topics.get(
                    main_topic.lower(), "related processes"
                )
                suggestion = template.format(
                    topic=main_topic,
                    related_topic=related_topic,
                    related_process=related_topic,
                    related_concept=related_topic,
                )
            else:
                suggestion = template.format(
                    topic=main_topic,
                    category="geological features",
                    location=main_topic,
                    phenomenon=main_topic,
                    specific_aspect="formation mechanisms",
                    related_comparison="similar processes",
                    alternative_scenario="different conditions",
                )

            suggestions.append(suggestion)

        return suggestions

    async def generate_response(
        self,
        query: str,
        context: Optional[Dict] = None,
        geological_data: Optional[Dict] = None,
    ) -> ConversationalResponse:
        """
        Generate conversational response for geological explanations

        Args:
            query: Natural language query
            context: Query context including intent and entities
            geological_data: Relevant geological data for the response

        Returns:
            ConversationalResponse with generated explanation
        """
        start_time = time.time()

        try:
            self.logger.info("Generating conversational response", query=query[:100])

            # Extract context information
            intent = context.get("intent") if context else None
            entities = context.get("entities", []) if context else []

            # Determine response type and generate appropriate response
            if intent == QueryIntent.EXPLANATION_REQUEST:
                response_text = self._generate_explanation_response(
                    query, entities, context
                )
                response_type = "explanation"
            elif intent == QueryIntent.COMPARISON_QUERY:
                response_text = self._generate_comparison_response(query, entities)
                response_type = "comparison"
            elif intent == QueryIntent.MONITORING_QUERY:
                response_text = self._generate_monitoring_response(
                    query, entities, geological_data
                )
                response_type = "monitoring"
            elif intent == QueryIntent.SIMULATION_REQUEST:
                response_text = self._generate_simulation_response(query, entities)
                response_type = "simulation"
            else:
                # Default explanation response
                response_text = self._generate_explanation_response(
                    query, entities, context
                )
                response_type = "explanation"

            # Extract main topic for follow-up suggestions
            main_topic = "geological processes"
            if entities:
                for entity in entities:
                    if entity.entity_type in [
                        EntityType.PROCESS,
                        EntityType.MATERIAL,
                        EntityType.LOCATION,
                    ]:
                        main_topic = entity.text
                        break

            # Generate follow-up suggestions
            follow_up_suggestions = self._generate_follow_up_suggestions(
                response_type, main_topic
            )

            # Determine confidence based on available knowledge
            confidence = (
                0.8
                if any(self._get_geological_knowledge(e.text) for e in entities)
                else 0.6
            )

            processing_time = (time.time() - start_time) * 1000

            self.logger.info(
                "Response generation completed",
                response_length=len(response_text),
                response_type=response_type,
            )

            return ConversationalResponse(
                response=response_text,
                response_type=response_type,
                confidence=confidence,
                sources=["geological_knowledge_base"],
                follow_up_suggestions=follow_up_suggestions,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            self.logger.error("Response generation failed", error=str(e))
            processing_time = (time.time() - start_time) * 1000

            return ConversationalResponse(
                response="I apologize, but I encountered an issue generating a response. Please try rephrasing your question about geological processes.",
                response_type="error",
                confidence=0.3,
                sources=[],
                follow_up_suggestions=[
                    "What geological process would you like to learn about?"
                ],
                processing_time_ms=processing_time,
            )
